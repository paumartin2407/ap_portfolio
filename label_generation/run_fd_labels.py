from __future__ import annotations

import argparse
import csv
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .fd_configs import get_fd_config


@dataclass(frozen=True)
class Instance:
    domain: str
    name: str
    domain_pddl: Path
    problem_pddl: Path


_SOLVED_PATTERNS = [
    re.compile(r"\bSolution found\b", re.IGNORECASE),
    re.compile(r"\bPlan length\b", re.IGNORECASE),
    re.compile(r"\bPlan cost\b", re.IGNORECASE),
]

_UNSOLVED_PATTERNS = [
    re.compile(r"\bNo solution\b", re.IGNORECASE),
    re.compile(r"\bSearch stopped without finding a solution\b", re.IGNORECASE),
    re.compile(r"\bdead end\b", re.IGNORECASE),
]


def iter_instances(data_dir: Path) -> Iterable[Instance]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    for domain_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        domain_pddl = domain_dir / "domain.pddl"
        if not domain_pddl.exists():
            continue

        for problem_pddl in sorted(domain_dir.glob("*.pddl")):
            if problem_pddl.name == "domain.pddl":
                continue
            yield Instance(
                domain=domain_dir.name,
                name=problem_pddl.name,
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
            )


def load_done_keys(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()

    done: set[tuple[str, str]] = set()
    with out_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Accept older files with extra columns.
        for row in reader:
            name = (row.get("name") or "").strip()
            domain = (row.get("domain") or "").strip()
            if name and domain:
                done.add((name, domain))
    return done


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_header_if_needed(out_csv: Path) -> None:
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return
    ensure_parent_dir(out_csv)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "domain", "solved", "time_seconds"])


def _limit_resources_preexec(mem_bytes: int | None) -> None:
    # Apply memory limit to the child process (and its descendants if they inherit it).
    # RLIMIT_AS is the most common on Linux; if unavailable, we silently skip.
    if mem_bytes is None:
        return
    try:
        import resource  # Linux/Unix only

        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        # Don't crash just because the limit couldn't be applied.
        return


def run_fd(
    fd_path: Path,
    solver_id: int,
    domain_pddl: Path,
    problem_pddl: Path,
    time_limit_s: int,
    mem_bytes: int | None,
    log_path: Path,
) -> tuple[int, float | float("nan"), str]:
    """Returns: (solved, time_seconds_or_nan, status_string)."""

    cfg = get_fd_config(solver_id)

    # Fast Downward driver-level resource limits.
    # - --overall-time-limit is in seconds.
    # - --overall-memory-limit is in MiB.
    driver_opts: list[str] = []
    if time_limit_s is not None and time_limit_s > 0:
        driver_opts += ["--overall-time-limit", str(int(time_limit_s))]
    if mem_bytes is not None and mem_bytes > 0:
        mem_mib = int(mem_bytes // (1024 * 1024))
        driver_opts += ["--overall-memory-limit", str(mem_mib)]

    # Correct ordering:
    # - driver options (overall limits, aliases) BEFORE domain/problem files
    # - evaluator/search options AFTER domain/problem files
    before_files = cfg.args if cfg.args_before_files else []
    after_files = [] if cfg.args_before_files else cfg.args
    cmd = [str(fd_path), *driver_opts, *before_files, str(domain_pddl), str(problem_pddl), *after_files]

    ensure_parent_dir(log_path)
    start = time.time()

    proc: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=(
                    (lambda: _limit_resources_preexec(mem_bytes)) if mem_bytes is not None else None
                ),
            )
            # Safety net: give the FD driver a bit of extra wall time to shut down cleanly.
            safety_timeout = int(time_limit_s) + 60 if time_limit_s is not None else None
            proc.communicate(timeout=safety_timeout)

        elapsed = time.time() - start
        solved = 1 if _log_indicates_solved(log_path) else 0
        if solved:
            return 1, elapsed, "solved"
        return 0, float("nan"), f"exit_{proc.returncode}"

    except subprocess.TimeoutExpired:
        _kill_child_process_group(proc)
        return 0, float("nan"), "timeout"
    except Exception as e:
        _kill_child_process_group(proc)
        return 0, float("nan"), f"error:{type(e).__name__}"


def _kill_child_process_group(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    try:
        # With start_new_session=True, pid is also the process group id.
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _log_indicates_solved(log_path: Path) -> bool:
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return False

    any_solved = any(p.search(text) for p in _SOLVED_PATTERNS)
    any_unsolved = any(p.search(text) for p in _UNSOLVED_PATTERNS)
    if any_solved and not any_unsolved:
        return True

    # Some FD builds may not print the exact phrases above.
    # As a fallback, accept common summary patterns.
    if re.search(r"\bsearch\s+exit\s+code\s*:\s*0\b", text, re.IGNORECASE) and any_solved:
        return True

    return False


def append_result(out_csv: Path, name: str, domain: str, solved: int, time_seconds: float) -> None:
    # Write one row and flush immediately for resume safety.
    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        time_field = "NaN" if (not solved or math.isnan(time_seconds)) else f"{time_seconds:.3f}"
        writer.writerow([name, domain, int(solved), time_field])
        f.flush()
        os.fsync(f.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fast Downward portfolio solver to generate labels.")
    parser.add_argument("--solver", type=int, required=True, help="Solver id 1-8 (see solvers.md)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument(
        "--fd-path",
        type=Path,
        required=True,
        help="Path to fast-downward.py (or wrapper script)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--time-limit", type=int, default=1800, help="Timeout per instance (seconds)")
    parser.add_argument("--mem-gb", type=float, default=8.0, help="Memory cap (GB) via RLIMIT_AS")
    parser.add_argument(
        "--no-mem-limit",
        action="store_true",
        help="Disable memory limiting (useful if RLIMIT_AS causes issues)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV instead of resuming",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional: only run this domain folder (e.g. agricola-sat18-strips)",
    )

    args = parser.parse_args(argv)

    fd_path: Path = args.fd_path
    if not fd_path.exists():
        raise FileNotFoundError(f"fast-downward.py not found: {fd_path}")

    out_csv: Path = args.out
    if args.overwrite and out_csv.exists():
        out_csv.unlink()

    write_header_if_needed(out_csv)

    done = load_done_keys(out_csv)

    mem_bytes = None
    if not args.no_mem_limit:
        mem_bytes = int(args.mem_gb * 1024 * 1024 * 1024)

    logs_root = Path(__file__).resolve().parent / "logs" / f"solver_{int(args.solver)}"

    total = 0
    skipped = 0
    solved_count = 0

    for inst in iter_instances(args.data_dir):
        if args.domain is not None and inst.domain != args.domain:
            continue

        total += 1
        key = (inst.name, inst.domain)
        if key in done:
            skipped += 1
            continue

        log_path = logs_root / inst.domain / f"{inst.name}.log"

        solved, tsec, status = run_fd(
            fd_path=fd_path,
            solver_id=int(args.solver),
            domain_pddl=inst.domain_pddl,
            problem_pddl=inst.problem_pddl,
            time_limit_s=int(args.time_limit),
            mem_bytes=mem_bytes,
            log_path=log_path,
        )

        if solved:
            solved_count += 1

        append_result(out_csv, inst.name, inst.domain, solved, tsec)
        done.add(key)

        print(
            f"[{args.solver}] {inst.domain}/{inst.name}: {status}"
            + (f" ({tsec:.2f}s)" if solved else ""),
            flush=True,
        )

    print(
        f"Done. considered={total}, skipped={skipped}, newly_run={total-skipped}, solved={solved_count}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
