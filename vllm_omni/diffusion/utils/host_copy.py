# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Overlapped device-to-host copies into pageable CPU tensors.

A direct ``cpu_tensor.copy_(cuda_tensor)`` into pageable memory runs at a
fraction of the DMA bandwidth and blocks both the host and the device stream
for its whole duration. Pipelines that decode a long video piece by piece
(for example one camera at a time) and gather the pieces into one CPU tensor
pay that stall once per piece.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch


def _drain(copied: torch.cuda.Event, staging: torch.Tensor, dst: torch.Tensor, inference: bool) -> None:
    copied.synchronize()
    # Helper threads do not inherit the caller's (thread-local) inference mode.
    with torch.inference_mode(inference):
        dst.copy_(staging)


class PinnedHostCopier:
    """Copy device tensors into CPU tensors through one reused pinned staging buffer.

    :meth:`copy` enqueues a non-blocking DMA into the staging buffer on the
    current stream and returns; a helper thread waits for it and copies the
    staging buffer into ``dst`` while the caller keeps launching device work.
    The next :meth:`copy` waits for the previous one to leave the staging
    buffer. Destinations are complete once :meth:`wait` returns or the
    context exits. Non-CUDA sources are copied synchronously. Every copy is a
    plain ``copy_``, so the destination bytes are exactly those of a direct copy.
    """

    def __init__(self, thread_name_prefix: str = "pinned-host-copy") -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._staging: torch.Tensor | None = None
        self._pending: Future[None] | None = None

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """``dst.copy_(src)`` for a CPU ``dst`` of ``src``'s shape and dtype, overlapped on CUDA."""
        if src.device.type != "cuda" or dst.device.type != "cpu" or dst.shape != src.shape or dst.dtype != src.dtype:
            self.wait()
            dst.copy_(src)
            return
        self.wait()
        staging = self._staging
        if staging is None or staging.shape != src.shape or staging.dtype != src.dtype:
            staging = self._staging = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
        staging.copy_(src, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record(torch.cuda.current_stream(src.device))
        self._pending = self._executor.submit(_drain, copied, staging, dst, torch.is_inference_mode_enabled())

    def wait(self) -> None:
        """Block until every enqueued copy has landed in its destination."""
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)
            self._staging = None

    def __enter__(self) -> PinnedHostCopier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["PinnedHostCopier"]
