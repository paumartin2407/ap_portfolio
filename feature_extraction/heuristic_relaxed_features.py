from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import statistics
import subprocess
import time
import re
import math

from feature_extraction.fd_limits import (
    FDMemoryLimitExceeded,
    looks_like_memory_limit,
    preexec_set_memory_limit,
)


def _run_capture(
    cmd: List[str],
    timeout_s: float,
    mem_limit_mb: Optional[int] = None,
) -> Tuple[str, float, bool, int]:
    """
    Returns: (combined_output, elapsed_seconds, timed_out, return_code)
    If timed out, return_code = -1.
    """
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=preexec_set_memory_limit(mem_limit_mb),
        )
        out = p.stdout or ""
        return out, (time.perf_counter() - t0), False, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else (
            e.stdout.decode("utf-8", "ignore") if e.stdout else ""
        )
        return out, (time.perf_counter() - t0), True, -1


def _run_capture_sas_stdin(
    cmd: List[str],
    sas_in: Path,
    timeout_s: float,
    mem_limit_mb: Optional[int] = None,
) -> Tuple[str, float, bool, int]:
    """Run a FD search-only command with SAS provided on stdin."""
    t0 = time.perf_counter()
    try:
        with sas_in.open("r", encoding="utf-8", errors="ignore") as fin:
            p = subprocess.run(
                cmd,
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
                preexec_fn=preexec_set_memory_limit(mem_limit_mb),
            )
        out = p.stdout or ""
        return out, (time.perf_counter() - t0), False, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else (
            e.stdout.decode("utf-8", "ignore") if e.stdout else ""
        )
        return out, (time.perf_counter() - t0), True, -1


def _first_int(patterns: List[str], text: str) -> int:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return -1


def _parse_goal_count_from_sas(sas_path: Optional[Path]) -> int:
    # SAS goal section:
    # begin_goal
    # <k>
    # <var val> lines...
    # end_goal
    if sas_path is None or not sas_path.exists():
        return -1
    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "begin_goal" and i + 1 < len(lines):
            try:
                return int(lines[i + 1].strip())
            except Exception:
                return -1
    return -1


def _parse_heuristic_value(text: str, key: str) -> float:
    """
    Extract heuristic value for evaluator name `key` from output.
    Returns inf if it says infinity, else float, else -1 if missing.
    """
    # Try a few common styles:
    patterns = [
        rf"initial\s+heuristic\s+value\s+for\s+{re.escape(key)}\s*:\s*([0-9]+|infinity)",
        rf"initial\s+h\s*\({re.escape(key)}\)\s*=\s*([0-9]+|infinity)",
        rf"{re.escape(key)}\s*initial\s*=\s*([0-9]+|infinity)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).lower()
            if val == "infinity":
                return math.inf
            try:
                return float(val)
            except Exception:
                return -1.0
    return -1.0


def _parse_preferred_ops(text: str) -> int:
    return _first_int([
        r"number\s+of\s+preferred\s+operators\s*:\s*(\d+)",
        r"preferred\s+operators\s*:\s*(\d+)",
    ], text)


def _parse_relaxed_plan_len(text: str) -> int:
    return _first_int([
        r"relaxed\s+plan\s+length\s*:\s*(\d+)",
        r"relaxed\s+plan\s+has\s+(\d+)\s+actions",
        r"length\s+of\s+relaxed\s+plan\s*:\s*(\d+)",
    ], text)


def _count_applicable_ops_prevail_only(sas_path: Path, init_vals: List[int]) -> int:
    """Count SAS operators whose prevail conditions hold in the initial state.

    This is a cheap, robust proxy for how many actions are immediately applicable.
    Note: conditional effect conditions are *not* required for applicability.
    """
    if not sas_path.exists():
        return -1

    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    nvars = len(init_vals)
    applicable = 0

    while i < len(lines):
        if lines[i].strip() == "begin_operator":
            # begin_operator
            # <name>
            # <num_prevail>
            # <var val> x num_prevail
            # <num_effects>
            # ...
            # end_operator
            try:
                num_prevail = int(lines[i + 2].strip())
            except Exception:
                return -1

            ok = True
            for k in range(num_prevail):
                try:
                    v, val = map(int, lines[i + 3 + k].split())
                except Exception:
                    ok = False
                    break
                if not (0 <= v < nvars) or init_vals[v] != val:
                    ok = False
                    break

            if ok:
                applicable += 1

            # jump to end_operator
            i = i + 3 + num_prevail
            while i < len(lines) and lines[i].strip() != "end_operator":
                i += 1

        i += 1

    return int(applicable)


def _parse_init_state_from_sas(sas_path: Path) -> Optional[List[int]]:
    # begin_state then one integer per variable
    if not sas_path.exists():
        return None
    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "begin_state":
            j = i + 1
            vals = []
            while j < len(lines) and lines[j].strip() != "end_state":
                try:
                    vals.append(int(lines[j].strip()))
                except Exception:
                    return None
                j += 1
            return vals
    return None


def _parse_operator_effects_for_delete_proxy(sas_path: Path, init_vals: List[int]) -> Tuple[float, float, float, float]:
    """
    Returns 4 proxies:
      - del_init_effect_fraction: init-overwrites / total effects
      - del_init_op_fraction: fraction of operators that overwrite at least one init value
      - del_init_effects_mean: mean #init-overwrites per operator
      - del_init_effects_max: max #init-overwrites in an operator
    """
    lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    total_effects = 0
    total_init_overwrites = 0
    per_op_overwrites = []

    nvars = len(init_vals)

    while i < len(lines):
        if lines[i].strip() == "begin_operator":
            # skip:
            # name, num_prevail, prevail lines, num_effects, effects...
            num_prevail = int(lines[i + 2].strip())
            j = i + 3 + num_prevail
            num_eff = int(lines[j].strip())
            j += 1

            op_over = 0
            for _ in range(num_eff):
                parts = list(map(int, lines[j].split()))
                # <num_cond> <condvar condval>... <eff_var> <from> <to>
                num_cond = parts[0]
                idx = 1 + 2 * num_cond
                eff_var = parts[idx]
                from_val = parts[idx + 1]
                # to_val = parts[idx + 2]  # not needed

                total_effects += 1
                if 0 <= eff_var < nvars and from_val == init_vals[eff_var]:
                    op_over += 1
                    total_init_overwrites += 1

                j += 1

            per_op_overwrites.append(op_over)

            # jump to end_operator
            i = j
            while i < len(lines) and lines[i].strip() != "end_operator":
                i += 1

        i += 1

    if not per_op_overwrites:
        return 0.0, 0.0, 0.0, 0.0

    del_init_effect_fraction = (total_init_overwrites / total_effects) if total_effects > 0 else 0.0
    del_init_op_fraction = sum(1 for x in per_op_overwrites if x > 0) / len(per_op_overwrites)
    del_init_effects_mean = float(statistics.mean(per_op_overwrites))
    del_init_effects_max = float(max(per_op_overwrites))

    return float(del_init_effect_fraction), float(del_init_op_fraction), float(del_init_effects_mean), float(del_init_effects_max)


@dataclass
class HeuristicProbeFeatures:
    probe_seconds: float

    h_ff_init: float
    h_add_init: float
    h_max_init: float

    helpful_actions_proxy: int  # preferred operators count (best-effort)
    relaxed_plan_length: int    # best-effort

    goal_count: int  # from SAS if available

    del_init_effect_fraction: float
    del_init_op_fraction: float
    del_init_effects_mean: float
    del_init_effects_max: float


def extract_heuristic_probe_features(
    fd_driver: Path,
    domain: Path,
    problem: Path,
    fd_search_bin: Optional[Path] = None,
    timeout_s: float = 3.0,
    sas_in: Optional[Path] = None,
    sas_out: Optional[Path] = None,
    log_path: Optional[Path] = None,
    mem_limit_mb: Optional[int] = None,
) -> HeuristicProbeFeatures:
    """
    Runs a very short FD probe intended to expose initial heuristic values and related signals.
    Uses fallbacks if some evaluators are unavailable in your build.

    Also computes a robust "delete interaction" proxy from SAS (if sas_out provided).
    """
    # Candidate commands: try richer evaluator set first, then fallback.
    candidates: List[List[str]] = []

    # Use unit-cost transforms so h_ff_init is a closer proxy for relaxed-plan length.
    # Note: some FD builds interpret `max()` as a *combining* evaluator requiring
    # an `evals` argument, not the hmax heuristic. We use `hmax()` instead when available.

    plan_file = None
    if log_path is not None:
        plan_file = str(Path(log_path).with_suffix(".plan"))

    can_search_only = (
        fd_search_bin is not None
        and Path(fd_search_bin).exists()
        and sas_in is not None
        and Path(sas_in).exists()
    )

    if can_search_only:
        # Search-only: run compiled `downward` on existing SAS (no translation).
        base = [str(fd_search_bin)]
        if plan_file is not None:
            base += ["--internal-plan-file", plan_file]

        candidates.append(base + [
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--evaluator", "hadd=add(transform=adapt_costs(one))",
            "--evaluator", "hhmax=hmax()",
            "--search", "lazy_greedy([hff,hadd,hhmax], preferred=[hff], cost_type=one)",
        ])

        candidates.append(base + [
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--evaluator", "hadd=add(transform=adapt_costs(one))",
            "--search", "lazy_greedy([hff,hadd], preferred=[hff], cost_type=one)",
        ])

        candidates.append(base + [
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--search", "lazy_greedy([hff], preferred=[hff], cost_type=one)",
        ])
    else:
        # Fallback: run `fast-downward.py` on domain+problem (will translate).
        candidates.append([
            str(fd_driver),
            str(domain), str(problem),
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--evaluator", "hadd=add(transform=adapt_costs(one))",
            "--evaluator", "hhmax=hmax()",
            "--search", "lazy_greedy([hff,hadd,hhmax], preferred=[hff], cost_type=one)",
        ])
        candidates.append([
            str(fd_driver),
            str(domain), str(problem),
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--evaluator", "hadd=add(transform=adapt_costs(one))",
            "--search", "lazy_greedy([hff,hadd], preferred=[hff], cost_type=one)",
        ])
        candidates.append([
            str(fd_driver),
            str(domain), str(problem),
            "--evaluator", "hff=ff(transform=adapt_costs(one))",
            "--search", "lazy_greedy([hff], preferred=[hff], cost_type=one)",
        ])

        # If user wants SAS output, include it for the probe run.
        if sas_out is not None:
            for c in candidates:
                c.insert(1, "--sas-file")
                c.insert(2, str(sas_out))

    out = ""
    elapsed = 0.0
    timed_out = False
    rc = -1

    chosen_cmd: Optional[List[str]] = None

    # Run first command that succeeds (rc==0), or at least times out (so we can parse partial output).
    for cmd in candidates:
        if can_search_only:
            out, elapsed, timed_out, rc = _run_capture_sas_stdin(
                cmd,
                Path(sas_in),
                timeout_s=timeout_s,
                mem_limit_mb=mem_limit_mb,
            )
        else:
            out, elapsed, timed_out, rc = _run_capture(cmd, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)

        if (not timed_out) and rc != 0 and looks_like_memory_limit(out, rc):
            raise FDMemoryLimitExceeded("FD heuristic probe hit memory limit")
        # Prefer a successful run; otherwise accept timeouts for partial signals.
        if rc == 0 or timed_out:
            chosen_cmd = cmd
            break
        # If it failed but still produced useful output, keep it as a fallback.
        if chosen_cmd is None and out.strip():
            chosen_cmd = cmd
            # but keep trying other candidates in case they succeed

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                f.write("# Heuristic probe (Fast Downward)\n")
                f.write(f"timeout_s: {timeout_s}\n")
                f.write(f"timed_out: {int(timed_out)}\n")
                f.write(f"return_code: {rc}\n")
                f.write(f"elapsed_s: {elapsed}\n\n")
                if chosen_cmd is not None:
                    f.write("command:\n")
                    f.write(" ".join(map(str, chosen_cmd)) + "\n\n")
                    if can_search_only and sas_in is not None:
                        f.write(f"stdin_sas: {sas_in}\n\n")
                else:
                    f.write("command: <none selected>\n\n")
                f.write("output:\n")
                f.write(out)
                if out and not out.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    # Parse heuristic values (may be missing depending on build/output).
    # Some FD versions print the *plugin* name (ff/add) instead of the alias (hff/hadd).
    h_ff = _parse_heuristic_value(out, "hff")
    if h_ff < 0:
        h_ff = _parse_heuristic_value(out, "ff")

    h_add = _parse_heuristic_value(out, "hadd")
    if h_add < 0:
        h_add = _parse_heuristic_value(out, "add")

    # We don't run an hmax evaluator portably here; keep as missing unless your FD prints one.
    h_max = _parse_heuristic_value(out, "hmax")
    if h_max < 0:
        h_max = _parse_heuristic_value(out, "max")

    helpful = _parse_preferred_ops(out)
    rplan_len = _parse_relaxed_plan_len(out)

    # If FD doesn't print relaxed plan length, fall back to the (unit-cost) FF value.
    if rplan_len < 0 and isinstance(h_ff, (int, float)) and h_ff >= 0 and math.isfinite(h_ff):
        rplan_len = int(h_ff)

    sas_for_stats = sas_in if (sas_in is not None and Path(sas_in).exists()) else sas_out
    goal_count = _parse_goal_count_from_sas(sas_for_stats) if sas_for_stats is not None else -1

    # Delete interaction proxy from SAS (robust, log-independent)
    del_frac = del_op_frac = del_mean = del_max = 0.0
    applicable_ops_init = -1
    if sas_for_stats is not None and Path(sas_for_stats).exists():
        init_vals = _parse_init_state_from_sas(Path(sas_for_stats))
        if init_vals is not None:
            del_frac, del_op_frac, del_mean, del_max = _parse_operator_effects_for_delete_proxy(Path(sas_for_stats), init_vals)
            applicable_ops_init = _count_applicable_ops_prevail_only(Path(sas_for_stats), init_vals)

    # If FD doesn't print preferred-op counts, fall back to a SAS-based applicability proxy.
    if helpful < 0 and applicable_ops_init >= 0:
        helpful = int(applicable_ops_init)

    return HeuristicProbeFeatures(
        probe_seconds=float(elapsed),
        h_ff_init=float(h_ff),
        h_add_init=float(h_add),
        h_max_init=float(h_max),
        helpful_actions_proxy=int(helpful),
        relaxed_plan_length=int(rplan_len),
        goal_count=int(goal_count),
        del_init_effect_fraction=float(del_frac),
        del_init_op_fraction=float(del_op_frac),
        del_init_effects_mean=float(del_mean),
        del_init_effects_max=float(del_max),
    )
