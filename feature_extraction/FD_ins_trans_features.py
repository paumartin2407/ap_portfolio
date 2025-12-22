import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

from feature_extraction.fd_limits import (
    FDMemoryLimitExceeded,
    looks_like_memory_limit,
    preexec_set_memory_limit,
)

@dataclass
class FDTranslationFeatures:
    translate_seconds: float
    sas_num_vars: int
    sas_var_domain_min: int
    sas_var_domain_mean: float
    sas_var_domain_max: int
    sas_total_facts_proxy: int  # sum of domain sizes (proxy for grounded fluents)
    sas_num_operators: int
    sas_op_prevail_min: int
    sas_op_prevail_mean: float
    sas_op_prevail_max: int
    sas_op_effects_min: int
    sas_op_effects_mean: float
    sas_op_effects_max: int
    sas_num_mutex_groups: int
    sas_mutex_size_min: int
    sas_mutex_size_mean: float
    sas_mutex_size_max: int
    sas_num_axioms: int


def _summarize_ints(xs: List[int]) -> Tuple[int, float, int]:
    if not xs:
        return 0, 0.0, 0
    return min(xs), float(statistics.mean(xs)), max(xs)


def run_fd_translate(
    fd_driver: Path,
    domain: Path,
    problem: Path,
    sas_out: Path,
    mem_limit_mb: Optional[int] = None,
) -> float:
    """
    Runs Fast Downward translation component and writes SAS to sas_out.
    Official usage supports --translate. :contentReference[oaicite:6]{index=6}
    """
    t0 = time.perf_counter()
    cmd = [str(fd_driver), "--translate", "--sas-file", str(sas_out), str(domain), str(problem)]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        preexec_fn=preexec_set_memory_limit(mem_limit_mb),
    )
    out = p.stdout or ""
    if p.returncode != 0:
        if looks_like_memory_limit(out, p.returncode):
            raise FDMemoryLimitExceeded("FD translate hit memory limit")
        raise RuntimeError(f"FD translate failed (rc={p.returncode})")
    return time.perf_counter() - t0


def parse_sas_file(sas_path: Path) -> FDTranslationFeatures:
    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0

    var_domain_sizes: List[int] = []
    mutex_sizes: List[int] = []
    op_prevails: List[int] = []
    op_effects: List[int] = []
    num_ops = 0
    num_axioms = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "begin_variable":
            # begin_variable
            # varX
            # <axiom layer or -1>
            # <domain size>
            # ... atoms ...
            # end_variable
            domain_size = int(lines[i + 3].strip())
            var_domain_sizes.append(domain_size)
            i += 4
            while i < len(lines) and lines[i].strip() != "end_variable":
                i += 1

        elif line == "begin_mutex_group":
            k = int(lines[i + 1].strip())
            mutex_sizes.append(k)
            i += 2 + k
            while i < len(lines) and lines[i].strip() != "end_mutex_group":
                i += 1

        elif line == "begin_operator":
            num_ops += 1
            # begin_operator
            # name
            # num_prevail
            # prevail lines
            # num_effects
            # effect lines
            # cost
            # end_operator
            num_prevail = int(lines[i + 2].strip())
            op_prevails.append(num_prevail)

            j = i + 3 + num_prevail
            num_eff = int(lines[j].strip())
            op_effects.append(num_eff)
            # skip effects + cost + end_operator
            i = j + 1 + num_eff
            while i < len(lines) and lines[i].strip() != "end_operator":
                i += 1

        elif line == "begin_rule":
            # Axioms / derived rules appear here.
            num_axioms += 1
            while i < len(lines) and lines[i].strip() != "end_rule":
                i += 1

        i += 1

    vmin, vmean, vmax = _summarize_ints(var_domain_sizes)
    mmin, mmean, mmax = _summarize_ints(mutex_sizes)
    pmin, pmean, pmax = _summarize_ints(op_prevails)
    emin, emean, emax = _summarize_ints(op_effects)

    return FDTranslationFeatures(
        translate_seconds=0.0,  # filled by caller
        sas_num_vars=len(var_domain_sizes),
        sas_var_domain_min=vmin,
        sas_var_domain_mean=vmean,
        sas_var_domain_max=vmax,
        sas_total_facts_proxy=sum(var_domain_sizes),
        sas_num_operators=num_ops,
        sas_op_prevail_min=pmin,
        sas_op_prevail_mean=pmean,
        sas_op_prevail_max=pmax,
        sas_op_effects_min=emin,
        sas_op_effects_mean=emean,
        sas_op_effects_max=emax,
        sas_num_mutex_groups=len(mutex_sizes),
        sas_mutex_size_min=mmin,
        sas_mutex_size_mean=mmean,
        sas_mutex_size_max=mmax,
        sas_num_axioms=num_axioms,
    )


def extract_fd_translation_features(
    fd_driver: Path,
    domain: Path,
    problem: Path,
    sas_out: Path,
    mem_limit_mb: Optional[int] = None,
) -> FDTranslationFeatures:
    secs = run_fd_translate(fd_driver, domain, problem, sas_out, mem_limit_mb=mem_limit_mb)
    feats = parse_sas_file(sas_out)
    feats.translate_seconds = secs
    return feats
