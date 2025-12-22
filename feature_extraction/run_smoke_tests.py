#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_instances(data_dir: Path) -> List[Tuple[Path, Path]]:
    items: List[Tuple[Path, Path]] = []
    for domain_folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        dom = domain_folder / "domain.pddl"
        if not dom.exists():
            continue
        problems = sorted(
            p
            for p in domain_folder.iterdir()
            if p.is_file() and p.suffix == ".pddl" and p.name != "domain.pddl"
        )
        for prob in problems:
            items.append((dom, prob))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run smoke_test_features on multiple instances and report per-run wall time."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Data directory containing domain folders (default: ./data)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of instances to run (default: 5)",
    )
    parser.add_argument(
        "--fd",
        type=Path,
        required=True,
        help="Path to fast-downward.py",
    )
    parser.add_argument(
        "--fd-search-bin",
        type=Path,
        default=None,
        help="Path to compiled downward binary (optional)",
    )
    parser.add_argument(
        "--sas-cache-dir",
        type=Path,
        default=REPO_ROOT / "sas_cache",
        help="SAS cache directory used by smoke_test_features (default: ./sas_cache)",
    )
    parser.add_argument(
        "--probe-heuristics-timeout",
        type=float,
        default=30.0,
        help="Heuristic probe timeout passed through to smoke_test_features",
    )
    parser.add_argument(
        "--probe-landmarks-timeout",
        type=float,
        default=30.0,
        help="Landmarks probe timeout passed through to smoke_test_features",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress smoke_test_features stdout/stderr",
    )

    args = parser.parse_args()

    instances = _iter_instances(args.data_dir)
    if not instances:
        raise FileNotFoundError(f"No instances found under {args.data_dir}")

    n = max(0, min(int(args.n), len(instances)))
    t0 = time.perf_counter()
    results: List[Tuple[str, str, float, int]] = []

    for i in range(n):
        dom, prob = instances[i]
        domain_name = dom.parent.name
        problem_name = prob.name

        cmd = [
            "python3",
            "-m",
            "feature_extraction.smoke_test_features",
            "--domain",
            str(dom),
            "--problem",
            str(prob),
            "--fd",
            str(args.fd),
            "--sas-cache-dir",
            str(args.sas_cache_dir),
            "--probe-heuristics-timeout",
            str(args.probe_heuristics_timeout),
            "--probe-landmarks-timeout",
            str(args.probe_landmarks_timeout),
        ]
        if args.fd_search_bin is not None:
            cmd += ["--fd-search-bin", str(args.fd_search_bin)]

        print(f"[{i+1}/{n}] {domain_name}/{problem_name}")
        t1 = time.perf_counter()
        if args.quiet:
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            p = subprocess.run(cmd)
        dt = time.perf_counter() - t1
        results.append((domain_name, problem_name, dt, p.returncode))
        print(f"    wall_s={dt:.2f} rc={p.returncode}")

    total = time.perf_counter() - t0
    print("\nSummary:")
    for domain_name, problem_name, dt, rc in results:
        print(f"- {domain_name}/{problem_name}\t{dt:.2f}s\trc={rc}")
    print(f"Total wall time: {total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
