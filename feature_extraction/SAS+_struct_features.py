from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import math
import statistics


def _entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def _percentile(sorted_xs: List[int], p: float) -> float:
    """Linear interpolation percentile in [0,1]. Input must be sorted."""
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_xs[int(k)])
    return sorted_xs[f] * (c - k) + sorted_xs[c] * (k - f)


def _summarize_ints(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {
            "min": 0.0, "mean": 0.0, "max": 0.0,
            "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0
        }
    xs_sorted = sorted(xs)
    return {
        "min": float(min(xs)),
        "mean": float(statistics.mean(xs)),
        "max": float(max(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "median": float(statistics.median(xs)),
        "p25": float(_percentile(xs_sorted, 0.25)),
        "p75": float(_percentile(xs_sorted, 0.75)),
    }


@dataclass
class SASStructureFeatures:
    # A) variables / domains (non-overlapping with FD_ins_trans_features)
    sas_domain_std: float
    sas_domain_median: float
    sas_domain_p25: float
    sas_domain_p75: float
    sas_domain_entropy: float

    # B) operators (non-overlapping with FD_ins_trans_features)
    op_prevail_std: float
    op_prevail_median: float
    op_effects_std: float
    op_effects_median: float
    op_mean_affected_vars: float
    op_mean_condition_vars: float
    op_affected_vars_fraction: float
    op_condition_vars_fraction: float
    op_mean_cond_to_eff_ratio: float

    # C) causal graph proxy (condition-var -> affected-var)
    cg_num_edges: int
    cg_edge_density: float
    cg_avg_out_degree: float
    cg_max_out_degree: int
    cg_avg_in_degree: float
    cg_max_in_degree: int
    cg_num_sources: int
    cg_num_sinks: int
    cg_num_isolated: int

    # D) DTG-style proxy: transition activity per variable
    dtg_trans_min: float
    dtg_trans_mean: float
    dtg_trans_max: float
    dtg_trans_std: float
    dtg_trans_median: float


def extract_sas_structure_features(sas_path: Path) -> SASStructureFeatures:
    """
    Extracts 40 robust SAS+ structure features from a Fast Downward SAS file.

    Uses only the stable SAS file structure:
      - begin_variable ... end_variable
      - begin_operator ... end_operator
    and parses operator effect lines to build:
      - causal graph proxy (condition var -> affected var)
      - per-variable transition activity proxy (DTG-ish).
    """
    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0

    # Parsed core objects
    var_domain_sizes: List[int] = []

    # For operators we capture: prevail count, effect count, condition vars, affected vars, transition vars.
    op_prevails: List[int] = []
    op_effect_counts: List[int] = []
    op_condition_var_counts: List[int] = []
    op_affected_var_counts: List[int] = []

    all_condition_vars: Set[int] = set()
    all_affected_vars: Set[int] = set()

    # We'll accumulate edges for the causal-graph proxy.
    cg_edges: Set[Tuple[int, int]] = set()

    # We'll fill this after we know nvars; temporarily store per-op transition vars.
    transition_vars_per_op: List[List[int]] = []

    # --- Parse SAS file ---
    while i < len(lines):
        line = lines[i].strip()

        if line == "begin_variable":
            # begin_variable
            # varX
            # <axiom layer or -1>
            # <domain size>
            domain_size = int(lines[i + 3].strip())
            var_domain_sizes.append(domain_size)
            i += 4
            while i < len(lines) and lines[i].strip() != "end_variable":
                i += 1

        elif line == "begin_operator":
            # begin_operator
            # name
            # num_prevail
            # prevail lines: "var value"
            # num_effects
            # effect lines: "<num_cond> <cond_var cond_val>... <eff_var> <from> <to>"
            num_prevail = int(lines[i + 2].strip())
            cond_vars: Set[int] = set()
            affected_vars: Set[int] = set()
            transition_vars: List[int] = []

            # prevail
            j = i + 3
            for _ in range(num_prevail):
                v = int(lines[j].split()[0])
                cond_vars.add(v)
                j += 1

            # effects
            num_eff = int(lines[j].strip())
            j += 1
            for _ in range(num_eff):
                parts = list(map(int, lines[j].split()))
                num_cond = parts[0]
                idx = 1
                for _c in range(num_cond):
                    cond_vars.add(parts[idx])
                    idx += 2  # skip cond value
                eff_var = parts[idx]
                affected_vars.add(eff_var)
                transition_vars.append(eff_var)
                j += 1

            # save operator stats
            op_prevails.append(num_prevail)
            op_effect_counts.append(num_eff)
            op_condition_var_counts.append(len(cond_vars))
            op_affected_var_counts.append(len(affected_vars))

            all_condition_vars |= cond_vars
            all_affected_vars |= affected_vars
            transition_vars_per_op.append(transition_vars)

            # causal graph edges from this operator (condition -> affected)
            for u in cond_vars:
                for v in affected_vars:
                    if u != v:
                        cg_edges.add((u, v))

            # jump to end_operator
            i = j
            while i < len(lines) and lines[i].strip() != "end_operator":
                i += 1

        i += 1

    # --- Compute derived stats ---
    nvars = len(var_domain_sizes)
    nops = len(op_prevails)

    dom_stats = _summarize_ints(var_domain_sizes)
    dom_entropy = _entropy_from_counts(var_domain_sizes)
    prev_stats = _summarize_ints(op_prevails)
    eff_stats = _summarize_ints(op_effect_counts)

    mean_affected = float(statistics.mean(op_affected_var_counts)) if op_affected_var_counts else 0.0
    mean_condition = float(statistics.mean(op_condition_var_counts)) if op_condition_var_counts else 0.0

    affected_frac = (len(all_affected_vars) / nvars) if nvars else 0.0
    condition_frac = (len(all_condition_vars) / nvars) if nvars else 0.0

    # mean(|cond_vars| / (|affected_vars|+eps))
    ratios = []
    for c, a in zip(op_condition_var_counts, op_affected_var_counts):
        ratios.append(c / (a + 1e-9))
    mean_ratio = float(statistics.mean(ratios)) if ratios else 0.0

    # causal graph proxy stats
    cg_num_edges = len(cg_edges)
    cg_edge_density = (cg_num_edges / (nvars * (nvars - 1))) if nvars > 1 else 0.0

    indeg = [0] * nvars
    outdeg = [0] * nvars
    for (u, v) in cg_edges:
        if 0 <= u < nvars and 0 <= v < nvars:
            outdeg[u] += 1
            indeg[v] += 1

    cg_avg_out = float(statistics.mean(outdeg)) if nvars else 0.0
    cg_max_out = int(max(outdeg)) if nvars else 0
    cg_avg_in = float(statistics.mean(indeg)) if nvars else 0.0
    cg_max_in = int(max(indeg)) if nvars else 0

    cg_sources = sum(1 for k in range(nvars) if indeg[k] == 0 and outdeg[k] > 0)
    cg_sinks = sum(1 for k in range(nvars) if indeg[k] > 0 and outdeg[k] == 0)
    cg_isolated = sum(1 for k in range(nvars) if indeg[k] == 0 and outdeg[k] == 0)

    # DTG-style proxy: transition activity per variable
    trans_per_var = [0] * nvars
    for trans_list in transition_vars_per_op:
        for v in trans_list:
            if 0 <= v < nvars:
                trans_per_var[v] += 1

    dtg_stats = _summarize_ints(trans_per_var)

    # --- Pack into dataclass (40 features) ---
    return SASStructureFeatures(
        # A (extras beyond FD_ins_trans_features)
        sas_domain_std=dom_stats["std"],
        sas_domain_median=dom_stats["median"],
        sas_domain_p25=dom_stats["p25"],
        sas_domain_p75=dom_stats["p75"],
        sas_domain_entropy=dom_entropy,

        # B (extras beyond FD_ins_trans_features)
        op_prevail_std=prev_stats["std"],
        op_prevail_median=prev_stats["median"],
        op_effects_std=eff_stats["std"],
        op_effects_median=eff_stats["median"],
        op_mean_affected_vars=mean_affected,
        op_mean_condition_vars=mean_condition,
        op_affected_vars_fraction=affected_frac,
        op_condition_vars_fraction=condition_frac,
        op_mean_cond_to_eff_ratio=mean_ratio,

        # C
        cg_num_edges=cg_num_edges,
        cg_edge_density=cg_edge_density,
        cg_avg_out_degree=cg_avg_out,
        cg_max_out_degree=cg_max_out,
        cg_avg_in_degree=cg_avg_in,
        cg_max_in_degree=cg_max_in,
        cg_num_sources=int(cg_sources),
        cg_num_sinks=int(cg_sinks),
        cg_num_isolated=int(cg_isolated),

        # D
        dtg_trans_min=dtg_stats["min"],
        dtg_trans_mean=dtg_stats["mean"],
        dtg_trans_max=dtg_stats["max"],
        dtg_trans_std=dtg_stats["std"],
        dtg_trans_median=dtg_stats["median"],
    )
