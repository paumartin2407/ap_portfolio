"""Fast Downward portfolio configs.

Solver IDs match `solvers.md`:

1) --alias lama-first
2) --alias seq-sat-lama-2011
3) FF greedy (lazy_greedy + preferred)
4) EHC with FF
5) CEA greedy
6) CG greedy
7) EHC with goalcount
8) weighted A* with FF
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
    1: FDConfig(1, "lama-first", ["--alias", "lama-first"], args_before_files=True),
    2: FDConfig(2, "seq-sat-lama-2011", ["--alias", "seq-sat-lama-2011"], args_before_files=True),
    3: FDConfig(
        3,
        "ff-gbfs",
        [
            "--evaluator",
            "hff=ff()",
            "--search",
            "lazy_greedy([hff], preferred=[hff])",
        ],
    ),
    4: FDConfig(
        4,
        "ehc-ff",
        [
            "--evaluator",
            "hff=ff()",
            "--search",
            "ehc(hff, preferred=[hff])",
        ],
    ),
    5: FDConfig(
        5,
        "cea-gbfs",
        [
            "--evaluator",
            "hcea=cea()",
            "--search",
            "lazy_greedy([hcea], preferred=[hcea])",
        ],
    ),
    6: FDConfig(
        6,
        "cg-gbfs",
        [
            "--evaluator",
            "hcg=cg()",
            "--search",
            "lazy_greedy([hcg])",
        ],
    ),
    7: FDConfig(
        7,
        "ehc-goalcount",
        [
            "--evaluator",
            "hgc=goalcount()",
            "--search",
            "ehc(hgc)",
        ],
    ),
    8: FDConfig(
        8,
        "wastar-ff",
        [
            "--evaluator",
            "hff=ff()",
            "--search",
            "eager_wastar([hff], w=5, preferred=[hff])",
        ],
    ),
}


def get_fd_config(solver_id: int) -> FDConfig:
    try:
        return FD_PORTFOLIO[int(solver_id)]
    except Exception as e:
        valid = ", ".join(str(k) for k in sorted(FD_PORTFOLIO))
        raise ValueError(f"Unknown solver id {solver_id}. Valid: {valid}") from e
