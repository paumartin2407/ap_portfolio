"""Fast Downward portfolio configs.

Solver IDs match `solvers.md`:

1) --alias lama-first
2) --alias seq-sat-fd-autotune-1
3) --alias seq-sat-fd-autotune-2
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FDConfig:
    solver_id: int
    name: str
    args: list[str]
    args_before_files: bool = False


FD_PORTFOLIO: dict[int, FDConfig] = {
    # FF
    1: FDConfig(
        1,
        "ff",
        [
            "--evaluator",
            "h=ff()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    ),

    # ADD
    2: FDConfig(
        2,
        "add",
        [
            "--evaluator",
            "h=add()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    ),

    # CG 
    3: FDConfig(
        3,
        "cg",
        [
            "--evaluator",
            "h=cg()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    )
}


def get_fd_config(solver_id: int) -> FDConfig:
    try:
        return FD_PORTFOLIO[int(solver_id)]
    except Exception as e:
        valid = ", ".join(str(k) for k in sorted(FD_PORTFOLIO))
        raise ValueError(f"Unknown solver id {solver_id}. Valid: {valid}") from e
