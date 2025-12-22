from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import resource


class FDMemoryLimitExceeded(RuntimeError):
    """Raised when a Fast Downward subprocess appears to have hit the memory cap."""


@dataclass(frozen=True)
class FDLimits:
    mem_limit_mb: Optional[int] = None

    @property
    def mem_limit_bytes(self) -> Optional[int]:
        if self.mem_limit_mb is None:
            return None
        mb = int(self.mem_limit_mb)
        if mb <= 0:
            return None
        return mb * 1024 * 1024


def preexec_set_memory_limit(mem_limit_mb: Optional[int]) -> Optional[Callable[[], None]]:
    """Return a preexec_fn that caps address space (RLIMIT_AS) for subprocesses.

    Linux-only / Unix-only.
    """
    limits = FDLimits(mem_limit_mb=mem_limit_mb)
    limit_bytes = limits.mem_limit_bytes
    if limit_bytes is None:
        return None

    def _set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return _set_limits


def looks_like_memory_limit(output: str, return_code: int) -> bool:
    """Best-effort detection of memory-limit / OOM failures.

    - If rlimit is hit, processes often terminate with SIGKILL (-9) or 137.
    - Some runs print explicit messages (bad_alloc, cannot allocate memory, etc.).
    """
    if return_code in (-9, 137):
        return True

    t = (output or "").lower()
    needles = [
        "memory limit exceeded",
        "out of memory",
        "std::bad_alloc",
        "bad alloc",
        "cannot allocate memory",
        "killed",
    ]
    return any(n in t for n in needles)
