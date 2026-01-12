from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedLog:
    solved: int
    plan_length: int | None
    plan_cost: int | None
    expanded: int | None
    reopened: int | None
    evaluated: int | None
    evaluations: int | None
    generated: int | None
    dead_ends: int | None
    search_time: float | None
    total_time: float | None


_SOLVED_RE = re.compile(r"\bSolution found\b", re.IGNORECASE)

_INT_METRICS: dict[str, re.Pattern[str]] = {
    "plan_length": re.compile(r"\bPlan length:\s*(\d+)\s*step\(s\)\.", re.IGNORECASE),
    "plan_cost": re.compile(r"\bPlan cost:\s*(\d+)\b", re.IGNORECASE),
    "expanded": re.compile(r"\bExpanded\s+(\d+)\s+state\(s\)\.", re.IGNORECASE),
    "reopened": re.compile(r"\bReopened\s+(\d+)\s+state\(s\)\.", re.IGNORECASE),
    "evaluated": re.compile(r"\bEvaluated\s+(\d+)\s+state\(s\)\.", re.IGNORECASE),
    "evaluations": re.compile(r"\bEvaluations:\s*(\d+)\b", re.IGNORECASE),
    "generated": re.compile(r"\bGenerated\s+(\d+)\s+state\(s\)\.", re.IGNORECASE),
    "dead_ends": re.compile(r"\bDead ends:\s*(\d+)\s+state\(s\)\.", re.IGNORECASE),
}

_FLOAT_METRICS: dict[str, re.Pattern[str]] = {
    "search_time": re.compile(r"\bSearch time:\s*([0-9]*\.?[0-9]+)s\b", re.IGNORECASE),
    "total_time": re.compile(r"\bTotal time:\s*([0-9]*\.?[0-9]+)s\b", re.IGNORECASE),
}


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_log_text(text: str) -> ParsedLog:
    solved = 1 if _SOLVED_RE.search(text) else 0

    if not solved:
        return ParsedLog(
            solved=0,
            plan_length=None,
            plan_cost=None,
            expanded=None,
            reopened=None,
            evaluated=None,
            evaluations=None,
            generated=None,
            dead_ends=None,
            search_time=None,
            total_time=None,
        )

    values: dict[str, Any] = {"solved": 1}
    for key, pat in _INT_METRICS.items():
        values[key] = _last_int(pat, text)
    for key, pat in _FLOAT_METRICS.items():
        values[key] = _last_float(pat, text)

    return ParsedLog(
        solved=1,
        plan_length=values["plan_length"],
        plan_cost=values["plan_cost"],
        expanded=values["expanded"],
        reopened=values["reopened"],
        evaluated=values["evaluated"],
        evaluations=values["evaluations"],
        generated=values["generated"],
        dead_ends=values["dead_ends"],
        search_time=values["search_time"],
        total_time=values["total_time"],
    )


def iter_log_files(logs_dir: Path):
    # Typical shape: logs/solver_1/<domain>/<problem>.pddl.log
    # We restrict to *.pddl.log and skip domain.pddl.log if present.
    for path in sorted(logs_dir.rglob("*.pddl.log")):
        if path.name == "domain.pddl.log":
            continue
        yield path


def _null(x: Any) -> str:
    return "" if x is None else str(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Fast Downward planner logs into a CSV of summary metrics.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("label_generation/logs"),
        help="Root logs directory containing solver_* folders (default: label_generation/logs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("solvers_labels/logs_metrics.csv"),
        help="Output CSV path (default: solvers_labels/logs_metrics.csv)",
    )

    args = parser.parse_args(argv)
    logs_dir: Path = args.logs_dir
    out_path: Path = args.out

    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "problem",
        "domain",
        "solver",
        "solved",
        "plan_length",
        "plan_cost",
        "expanded",
        "reopened",
        "evaluated",
        "evaluations",
        "generated",
        "dead_ends",
        "search_time",
        "total_time",
    ]

    rows_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for log_path in iter_log_files(logs_dir):
            rel = log_path.relative_to(logs_dir)
            parts = rel.parts
            if len(parts) < 3:
                # Unexpected structure; skip rather than crash.
                continue

            solver_dir = parts[0]
            domain = parts[1]

            m = re.match(r"solver_(\d+)$", solver_dir)
            if not m:
                # Unexpected solver directory name.
                continue
            solver_id = int(m.group(1))

            # Convert p01.pddl.log -> p01.pddl
            problem = log_path.name[: -len(".log")] if log_path.name.endswith(".log") else log_path.name

            try:
                text = log_path.read_text(errors="replace")
            except Exception:
                # If the file can't be read, treat as unsolved with null metrics.
                parsed = ParsedLog(
                    solved=0,
                    plan_length=None,
                    plan_cost=None,
                    expanded=None,
                    reopened=None,
                    evaluated=None,
                    evaluations=None,
                    generated=None,
                    dead_ends=None,
                    search_time=None,
                    total_time=None,
                )
            else:
                parsed = parse_log_text(text)

            writer.writerow(
                {
                    "problem": problem,
                    "domain": domain,
                    "solver": solver_id,
                    "solved": parsed.solved,
                    "plan_length": _null(parsed.plan_length),
                    "plan_cost": _null(parsed.plan_cost),
                    "expanded": _null(parsed.expanded),
                    "reopened": _null(parsed.reopened),
                    "evaluated": _null(parsed.evaluated),
                    "evaluations": _null(parsed.evaluations),
                    "generated": _null(parsed.generated),
                    "dead_ends": _null(parsed.dead_ends),
                    "search_time": _null(parsed.search_time),
                    "total_time": _null(parsed.total_time),
                }
            )
            rows_written += 1

    print(f"Wrote {rows_written} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
