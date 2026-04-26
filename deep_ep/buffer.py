import os
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

# Phantora: when running under simulation, the C++ runtime is never invoked
# (every dispatch/combine path is short-circuited to a synthetic flow), so we
# tolerate `deep_ep_cpp` being absent. Outside Phantora, import as upstream.
_PHANTORA = os.environ.get('PHANTORA') == '1'
try:
    # noinspection PyUnresolvedReferences
    import deep_ep_cpp
    # noinspection PyUnresolvedReferences
    from deep_ep_cpp import Config, EventHandle
except ImportError:
    if not _PHANTORA:
        raise
    deep_ep_cpp = None

    class Config:  # type: ignore
        """Phantora stub. Real Config is unused under simulation."""
        def __init__(self, *args, **kwargs):
            pass

    class EventHandle:  # type: ignore
        """Phantora stub. Real EventHandle is unused under simulation."""
        def current_stream_wait(self):
            pass

from .utils import EventOverlap, check_nvlink_connections


# Phantora simulation helpers ------------------------------------------------

def _phantora_owner_rank(num_experts: int, num_ranks: int) -> torch.Tensor:
    """Return CPU int64 tensor `[num_experts]` mapping global expert id → owner
    rank. Mirrors vLLM's `determine_expert_map` partitioning: round-robin with
    the remainder all going to the last rank."""
    local = num_experts // num_ranks
    if local == 0:
        # Fewer experts than ranks: every expert lives on rank 0..num_experts-1
        owner = torch.arange(num_experts, dtype=torch.int64)
        return owner
    owner = torch.arange(num_experts, dtype=torch.int64) // local
    # Last rank absorbs the remainder
    owner.clamp_(max=num_ranks - 1)
    return owner


def _phantora_sendcounts(topk_idx: torch.Tensor, num_experts: int,
                         num_ranks: int) -> torch.Tensor:
    """Per-peer token count this rank sends. Each token is sent once per
    distinct destination rank in its topk (matching DeepEP semantics)."""
    owner = _phantora_owner_rank(num_experts, num_ranks)
    # Map invalid (-1) to a sentinel so they don't count
    safe = topk_idx.detach().to('cpu', dtype=torch.int64)
    valid = safe.ge(0) & safe.lt(num_experts)
    safe = safe.clamp(0, num_experts - 1)
    dest = owner[safe]                              # [num_tokens, topk]
    dest = dest.where(valid, torch.full_like(dest, -1))
    sendcounts = torch.zeros(num_ranks, dtype=torch.int64)
    for r in range(num_ranks):
        sendcounts[r] = (dest == r).any(dim=1).sum()
    return sendcounts


def _phantora_pick_device_group(group: dist.ProcessGroup) -> dist.ProcessGroup:
    """Return a NCCL-capable group equivalent to `group`.

    vLLM constructs `deep_ep.Buffer(group=cpu_group)` — a gloo-backed group
    used only for object exchange. Wire ops (dist.batch_isend_irecv with CUDA
    tensors) need a NCCL backend so Phantora's libnccl shim can intercept
    them. If `group` already has NCCL we use it; otherwise look up the EP
    device_group via vLLM (lazy import to keep deepep usable outside vLLM).
    """
    try:
        if dist.get_backend(group) == 'nccl':
            return group
    except (RuntimeError, ValueError):
        pass
    try:
        from vllm.distributed import get_ep_group
        return get_ep_group().device_group
    except Exception:
        return group


def _phantora_submit_flows(group: dist.ProcessGroup, sendcounts: torch.Tensor,
                            recvcounts: torch.Tensor, hidden: int,
                            local_rank: int, num_ranks: int) -> None:
    """Submit per-peer dummy bf16 transfers via dist.batch_isend_irecv. Phantora
    intercepts the underlying ncclSend/ncclRecv pairs and models per-pair flows
    in the simulator. Bytes never actually move."""
    if num_ranks <= 1:
        return
    group = _phantora_pick_device_group(group)
    # Allocate one fake bf16 buffer per peer with the right element count.
    ops = []
    for r in range(num_ranks):
        if r == local_rank:
            continue
        s_n = int(sendcounts[r].item())
        r_n = int(recvcounts[r].item())
        if s_n > 0:
            send_buf = torch.empty(s_n * hidden, dtype=torch.bfloat16, device='cuda')
            ops.append(dist.P2POp(dist.isend, send_buf, r, group=group))
        if r_n > 0:
            recv_buf = torch.empty(r_n * hidden, dtype=torch.bfloat16, device='cuda')
            ops.append(dist.P2POp(dist.irecv, recv_buf, r, group=group))
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


def _phantora_exchange_sendcounts(sendcounts: torch.Tensor,
                                   group: dist.ProcessGroup,
                                   num_ranks: int) -> torch.Tensor:
    """All-gather sendcounts so this rank knows how many tokens each peer is
    sending it. Returns recvcounts [num_ranks] (recvcounts[r] = bytes from r).

    Stays on the gloo (CPU) group on purpose — under Phantora the NCCL
    AllGather is intercepted and returns uninitialised garbage (the shim
    only models bytes-on-the-wire, it doesn't actually transfer the data).
    Sendcounts are a tiny int64 metadata tensor (num_ranks² ints), so a
    real gloo AllGather over OS-level transport is essentially free and
    gives us correct values for downstream `recv_buf` sizing — getting
    the real numbers here is what stops `torch.empty(r_n * hidden, ...)`
    blowing up with "tried to allocate 366 EB" when r_n is garbage."""
    # If the caller's `group` isn't gloo, find one — vLLM passes us its
    # cpu_group at Buffer.__init__ which is what we want here.
    cpu_group = group
    try:
        if dist.get_backend(group) != 'gloo':
            from vllm.distributed import get_ep_group
            cpu_group = get_ep_group().cpu_group
    except Exception:
        pass

    sendcounts_cpu = sendcounts.detach().cpu().contiguous()
    all_sendcounts = torch.zeros(num_ranks * num_ranks, dtype=torch.int64)
    dist.all_gather_into_tensor(all_sendcounts, sendcounts_cpu, group=cpu_group)
    return all_sendcounts.view(num_ranks, num_ranks)


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(self, group: Optional[dist.ProcessGroup],
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 explicitly_destroy: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        # Phantora: skip nvlink topology probing (queries pynvml, which has no
        # meaningful answer under the simulated CUDA shim).
        if not _PHANTORA:
            check_nvlink_connections(group)

        # Initialize the CPP runtime
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy

        # Phantora: skip the entire CPP runtime initialisation. Every dispatch/
        # combine path checks _PHANTORA and synthesises outputs on CPU, so the
        # runtime is never accessed. Returning here avoids cudaIpc / nvshmem
        # calls that would either no-op silently (corrupting state) or fail.
        if _PHANTORA:
            self.runtime = None
            return

        self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # Enable IBGDA
            assert num_qps_per_rank > 0
            os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
            os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            os.environ['NVSHMEM_QP_DEPTH'] = os.environ.get('NVSHMEM_QP_DEPTH', '1024')

            # Reduce gpu memory usage
            # 6 default teams + 1 extra team
            os.environ['NVSHMEM_MAX_TEAMS'] = '7'
            # Disable NVLink SHArP
            os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
            # NOTES: NVSHMEM initialization requires at least 256 MiB
            os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

            if not allow_mnnvl:
                # Disable multi-node NVLink detection
                os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'

            # Synchronize using the root ID
            if (low_latency_mode and self.rank == 0) or (not low_latency_mode and self.runtime.get_rdma_rank() == 0):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            nvshmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = nvshmem_unique_ids[0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        if _PHANTORA:
            self.runtime = None
            return

        self.runtime.destroy()
        self.runtime = None

    # ----- Phantora simulation shims -------------------------------------

    def _phantora_dispatch(self, x, handle, topk_idx, topk_weights,
                           num_tokens_per_rank, is_token_in_rank,
                           num_tokens_per_expert, expert_alignment):
        """Synthesise dispatch outputs and submit per-pair flows. Inputs follow
        the upstream signatures of `dispatch` / `internode_dispatch`."""
        x_tensor, x_scales = x if isinstance(x, tuple) else (x, None)
        device = x_tensor.device
        hidden = x_tensor.size(1)
        num_ranks = self.group_size
        num_experts = (num_tokens_per_expert.numel()
                       if num_tokens_per_expert is not None
                       else (topk_idx.max().item() + 1 if topk_idx is not None else 0))

        # Cached path (handle != None): reuse stored counts. Otherwise compute.
        if handle is not None:
            sendcounts, recvcounts, all_sendcounts, src_topk_idx, src_topk_weights, \
                stored_num_experts, stored_local_experts = handle
            num_experts = stored_num_experts
        else:
            assert topk_idx is not None
            sendcounts = _phantora_sendcounts(topk_idx, num_experts, num_ranks)
            all_sendcounts = _phantora_exchange_sendcounts(sendcounts, self.group, num_ranks)
            recvcounts = all_sendcounts[:, self.rank]
            src_topk_idx = topk_idx
            src_topk_weights = topk_weights

        # Submit per-pair fake flows. Phantora intercepts ncclSend/Recv.
        _phantora_submit_flows(self.group, sendcounts, recvcounts,
                               hidden, self.rank, num_ranks)

        num_recv_tokens = int(recvcounts.sum().item())
        local_num_experts = max(1, num_experts // num_ranks)

        recv_x = torch.empty(num_recv_tokens, hidden,
                             dtype=x_tensor.dtype, device=device)
        recv_x_scales = (torch.empty(num_recv_tokens, x_scales.size(1),
                                      dtype=x_scales.dtype, device=device)
                         if x_scales is not None else None)
        topk = src_topk_idx.size(1) if src_topk_idx is not None else 1
        recv_topk_idx = torch.empty(num_recv_tokens, topk,
                                    dtype=torch.int64, device=device)
        recv_topk_weights = torch.empty(num_recv_tokens, topk,
                                        dtype=torch.float32, device=device)
        # vLLM's _do_dispatch later remaps -1 entries; fill with valid local
        # expert indices so downstream code (including the remapping branch)
        # sees in-range values.
        if num_recv_tokens > 0:
            recv_topk_idx.fill_(0)
            recv_topk_weights.fill_(1.0 / topk)

        num_recv_tokens_per_expert_list = [
            num_recv_tokens // local_num_experts
        ] * local_num_experts
        # Distribute remainder across the first few experts
        rem = num_recv_tokens - sum(num_recv_tokens_per_expert_list)
        for i in range(rem):
            num_recv_tokens_per_expert_list[i] += 1

        new_handle = (sendcounts, recvcounts, all_sendcounts,
                      src_topk_idx, src_topk_weights,
                      num_experts, local_num_experts)

        out_x = (recv_x, recv_x_scales) if x_scales is not None else recv_x
        return (out_x, recv_topk_idx, recv_topk_weights,
                num_recv_tokens_per_expert_list, new_handle, EventOverlap(None))

    def _phantora_combine(self, x, handle, topk_weights):
        """Synthesise combine output and submit reverse-direction flows."""
        sendcounts, recvcounts, all_sendcounts, src_topk_idx, src_topk_weights, \
            num_experts, local_num_experts = handle
        num_ranks = self.group_size
        # Combine reverses dispatch direction: receive what we previously sent.
        _phantora_submit_flows(self.group, recvcounts, sendcounts,
                               x.size(1), self.rank, num_ranks)
        # Output token count is the original pre-dispatch token count =
        # the number of rows in src_topk_idx.
        orig_num_tokens = src_topk_idx.size(0) if src_topk_idx is not None else x.size(0)
        combined_x = torch.empty(orig_num_tokens, x.size(1),
                                  dtype=x.dtype, device=x.device)
        combined_topk_weights = (
            torch.empty(orig_num_tokens, src_topk_weights.size(1),
                        dtype=torch.float32, device=x.device)
            if src_topk_weights is not None else None)
        return combined_x, combined_topk_weights, EventOverlap(None)

    def _phantora_low_latency_dispatch(self, x, topk_idx,
                                        num_max_dispatch_tokens_per_rank,
                                        num_experts, use_fp8, return_recv_hook):
        """LL-mode dispatch shim. Submits flows; returns shape-correct buffers
        sized to the per-rank max-dispatch ceiling."""
        device = x.device
        hidden = x.size(1)
        num_ranks = self.group_size
        sendcounts = _phantora_sendcounts(topk_idx, num_experts, num_ranks)
        all_sendcounts = _phantora_exchange_sendcounts(sendcounts, self.group, num_ranks)
        recvcounts = all_sendcounts[:, self.rank]
        _phantora_submit_flows(self.group, sendcounts, recvcounts,
                               hidden, self.rank, num_ranks)

        local_num_experts = max(1, num_experts // num_ranks)
        cap = num_max_dispatch_tokens_per_rank * num_ranks
        if use_fp8:
            packed_recv_x = torch.empty(local_num_experts, cap, hidden,
                                         dtype=torch.float8_e4m3fn, device=device)
            packed_recv_x_scales = torch.empty(local_num_experts, cap,
                                                hidden // 128,
                                                dtype=torch.float32,
                                                device=device)
            recv_x_out = (packed_recv_x, packed_recv_x_scales)
        else:
            packed_recv_x = torch.empty(local_num_experts, cap, hidden,
                                         dtype=torch.bfloat16, device=device)
            recv_x_out = packed_recv_x
        packed_recv_count = torch.zeros(local_num_experts,
                                         dtype=torch.int32, device=device)
        # Distribute received tokens evenly across local experts.
        total_recv = int(recvcounts.sum().item())
        per_expert = total_recv // local_num_experts
        for i in range(local_num_experts):
            packed_recv_count[i] = per_expert + (1 if i < total_recv - per_expert * local_num_experts else 0)

        # Handle for combine; carry orig_num_tokens (= topk_idx.shape[0]) so
        # combine can rebuild the output shape.
        handle = (None, None, num_max_dispatch_tokens_per_rank,
                  hidden, num_experts, topk_idx.size(0))

        hook = (lambda: None) if return_recv_hook else None
        return recv_x_out, packed_recv_count, handle, EventOverlap(None), hook

    def _phantora_low_latency_combine(self, x, topk_idx, topk_weights, handle,
                                       out, return_recv_hook):
        """LL-mode combine shim."""
        _, _, num_max_dispatch_tokens_per_rank, hidden, num_experts, orig_num_tokens = handle
        num_ranks = self.group_size
        # Fake reverse-direction flow: assume balanced under uniform routing.
        sendcounts = torch.full((num_ranks,),
                                 max(1, orig_num_tokens // num_ranks),
                                 dtype=torch.int64)
        _phantora_submit_flows(self.group, sendcounts, sendcounts,
                               hidden, self.rank, num_ranks)
        if out is not None:
            combined_x = out
        else:
            combined_x = torch.empty(orig_num_tokens, hidden,
                                      dtype=torch.bfloat16, device=x.device)
        hook = (lambda: None) if return_recv_hook else None
        return combined_x, EventOverlap(None), hook


    @staticmethod
    def is_sm90_compiled():
        if _PHANTORA:
            return True
        return deep_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        if _PHANTORA:
            # Rough upper-bound formula consistent with DeepEP's docs. Buffer
            # size is irrelevant under simulation (we never allocate it).
            return num_max_dispatch_tokens_per_rank * hidden * 2 * num_ranks
        return deep_ep_cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        if _PHANTORA:
            return torch.cuda.current_stream()
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    def get_local_buffer_tensor(self, dtype: torch.dtype, size: Optional[torch.Size] = None,
                                offset: int = 0, use_rdma_buffer: bool = False) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[:size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 32, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        if _PHANTORA:
            # Compute layout entirely on CPU from the synthetic topk indices.
            # No comm cost — vLLM uses these counts only for CPU-side bookkeeping
            # before calling dispatch (which is where wire flows are submitted).
            num_ranks = self.group_size
            owner = _phantora_owner_rank(num_experts, num_ranks)
            safe = topk_idx.detach().to('cpu', dtype=torch.int64)
            valid = safe.ge(0) & safe.lt(num_experts)
            safe = safe.clamp(0, num_experts - 1)
            dest = owner[safe].where(valid, torch.full_like(safe, -1))
            num_tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int32)
            for r in range(num_ranks):
                num_tokens_per_rank[r] = (dest == r).any(dim=1).sum()
            num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32)
            flat_valid = safe[valid]
            if flat_valid.numel() > 0:
                num_tokens_per_expert.scatter_add_(
                    0, flat_valid, torch.ones_like(flat_valid, dtype=torch.int32))
            is_token_in_rank = torch.zeros(
                topk_idx.shape[0], num_ranks, dtype=torch.bool)
            for r in range(num_ranks):
                is_token_in_rank[:, r] = (dest == r).any(dim=1)
            device = topk_idx.device
            return (num_tokens_per_rank.to(device), None,
                    num_tokens_per_expert.to(device),
                    is_token_in_rank.to(device), EventOverlap(None))

        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
            self.runtime.get_dispatch_layout(topk_idx, num_experts, getattr(previous_event, 'event', None),
                                             async_finish, allocate_on_comm_stream)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1, num_worst_tokens: int = 0,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Phantora: synthesise dispatch outputs and submit per-pair flows. Same
        # code path for intranode and internode — we don't distinguish.
        if _PHANTORA:
            return self._phantora_dispatch(
                x, handle, topk_idx, topk_weights, num_tokens_per_rank,
                is_token_in_rank, num_tokens_per_expert, expert_alignment)

        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            assert num_worst_tokens == 0, 'Internode dispatch does not support `num_worst_tokens > 0`'
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                                           topk_idx, topk_weights, expert_alignment, config, previous_event, async_finish, allocate_on_comm_stream)

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None,
                None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,
                expert_alignment, num_worst_tokens, config,
                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(event)

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Phantora: synthesise combine output and submit reverse-direction flows.
        if _PHANTORA:
            return self._phantora_combine(x, handle, topk_weights)

        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)

        # NOTES: the second `_` is for the sending side, so we should use the third one
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
            x, topk_weights, bias_0, bias_1,
            src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, config,
            getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        if _PHANTORA:
            return self._phantora_dispatch(
                x, handle, topk_idx, topk_weights, num_tokens_per_rank,
                is_token_in_rank, num_tokens_per_expert, expert_alignment)
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                None, None, is_token_in_rank, None,
                num_recv_tokens, num_rdma_recv_tokens,
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                0, 0, None, None, None, None,
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (is_token_in_rank,
                      rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                      recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                      recv_src_meta, send_rdma_head, send_nvl_head)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        if _PHANTORA:
            return self._phantora_combine(x, handle, topk_weights)
        assert config is not None

        # Unpack handle and bias
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(
            x, topk_weights, bias_0, bias_1,
            src_meta, is_combined_token_in_rank,
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
            send_rdma_head, send_nvl_head, config, getattr(previous_event, 'event', None),
            async_finish, allocate_on_comm_stream)
        return combined_x, combined_topk_weights, EventOverlap(event)

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        if _PHANTORA:
            return
        self.runtime.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            dispatch_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and precisely localizing slow anomalies.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        if _PHANTORA:
            return self._phantora_low_latency_dispatch(
                x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                use_fp8, return_recv_hook)

        packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook = \
            self.runtime.low_latency_dispatch(x, topk_idx,
                                              cumulative_local_expert_recv_stats,
                                              dispatch_wait_recv_cost_stats,
                                              num_max_dispatch_tokens_per_rank, num_experts,
                                              use_fp8, round_scale, use_ue8m0,
                                              async_finish, return_recv_hook)
        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1), num_experts)
        tensors_to_record = (x, topk_idx,
                             packed_recv_x, packed_recv_x_scales, packed_recv_count,
                             packed_recv_src_info, packed_recv_layout_range,
                             cumulative_local_expert_recv_stats)
        return (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x, packed_recv_count, handle, \
            EventOverlap(event, tensors_to_record if async_finish else None), hook

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.
            combine_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        if _PHANTORA:
            return self._phantora_low_latency_combine(
                x, topk_idx, topk_weights, handle, out, return_recv_hook)
        combined_x, event, hook = self.runtime.low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range,
                                                                   combine_wait_recv_cost_stats,
                                                                   num_max_dispatch_tokens_per_rank, num_experts,
                                                                   use_logfmt, zero_copy, async_finish, return_recv_hook,
                                                                   out)
        tensors_to_record = (x, topk_idx, topk_weights, src_info, layout_range, combined_x)
        return combined_x, EventOverlap(event, tensors_to_record if async_finish else None), hook

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        return self.runtime.get_next_low_latency_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
