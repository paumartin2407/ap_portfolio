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
    1: FDConfig(1, "lama-first", ["--alias", "lama-first"], args_before_files=True),
    2: FDConfig(
        2,
        "autotune-1-phase1",
        [
            "--search",
            "let(hff,ff(transform=adapt_costs(one)),"
            "lazy(alt([single(sum([g(),weight(hff,10)])),"
            "single(sum([g(),weight(hff,10)]),pref_only=true)],boost=2000),"
            "preferred=[hff],reopen_closed=false,cost_type=one))",
        ],
        args_before_files=False,  # <-- IMPORTANT: --search must go after domain/problem files
    ),
    3: FDConfig(
        3,
        "autotune-2-phase2",
        [
            "--search",
            "let(hcea,cea(transform=adapt_costs(plusone)),"
            "ehc(hcea,preferred=[hcea],preferred_usage=prune_by_preferred,cost_type=normal))",
        ],
        args_before_files=False,  # <-- critical: --search must go AFTER domain/problem files
    ),
    4: FDConfig(
        4,
        "autotune-2-phase2",
        [
            "--search",
            "let(hcea,cea(transform=adapt_costs(plusone)),"
            "let(hcg,cg(transform=adapt_costs(one)),"
            "let(hgc,goalcount(transform=adapt_costs(plusone)),"
            "let(hff,ff(),"
            "lazy(alt(["
            "single(sum([weight(g(),2),weight(hff,3)])),"
            "single(sum([weight(g(),2),weight(hff,3)]),pref_only=true),"
            "single(sum([weight(g(),2),weight(hcg,3)])),"
            "single(sum([weight(g(),2),weight(hcg,3)]),pref_only=true),"
            "single(sum([weight(g(),2),weight(hcea,3)])),"
            "single(sum([weight(g(),2),weight(hcea,3)]),pref_only=true),"
            "single(sum([weight(g(),2),weight(hgc,3)])),"
            "single(sum([weight(g(),2),weight(hgc,3)]),pref_only=true)"
            "],boost=200),"
            "preferred=[hcea,hgc],reopen_closed=false,cost_type=one)"
            "))))",
        ],
        args_before_files=False,  # <-- critical ordering in your runner
    ),
    # FF ONLY (fast greedy best-first, no landmarks)
    5: FDConfig(
        5,
        "ff",
        [
            "--evaluator",
            "h=ff()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    ),

    # CEA ONLY (greedy best-first guided by context-enhanced additive)
    6: FDConfig(
        6,
        "cea",
        [
            "--evaluator",
            "h=cea()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    ),

    # EHC with FF (hill-climbing variant guided by FF)
    7: FDConfig(
        7,
        "add",
        [
            "--evaluator",
            "h=add()",
            "--search",
            "lazy_greedy([h], preferred=[h])",
        ],
        args_before_files=False,
    ),

    # EHC with FF (hill-climbing variant guided by FF)
    8: FDConfig(
        8,
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
