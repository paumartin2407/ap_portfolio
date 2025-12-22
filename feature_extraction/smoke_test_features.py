#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import importlib.util
import time

# Allow running this file directly without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.PDDL_syn_features import extract_pddl_syntactic  # noqa: E402
from feature_extraction.FD_ins_trans_features import (  # noqa: E402
    extract_fd_translation_features,
    parse_sas_file,
)

# Optional probe feature modules (may not exist / may be experimental).
try:
    from feature_extraction.heuristic_relaxed_features import extract_heuristic_probe_features
except Exception:  # pragma: no cover
    extract_heuristic_probe_features = None

try:
    from feature_extraction.landmarks_features import extract_landmark_probe_features
except Exception:  # pragma: no cover
    extract_landmark_probe_features = None


def _try_load_sas_plus_extractor():
    """Load SAS+ extractor from `feature_extraction/SAS+_struct_features.py`.

    The '+' in the filename makes it non-importable as a normal Python module,
    so we use a dynamic import by file path.
    """
    sas_plus_path = REPO_ROOT / "feature_extraction" / "SAS+_struct_features.py"
    if not sas_plus_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("sas_plus_struct_features", sas_plus_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Python 3.12+ `dataclasses` expects the module to be present in `sys.modules`.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "extract_sas_structure_features", None)


def _pick_default_instance(data_dir: Path) -> tuple[Path, Path]:
    # Pick a deterministic, small-ish looking instance.
    # Prefer agricola p01 if present.
    candidate = data_dir / "agricola-sat18-strips"
    if (candidate / "domain.pddl").exists() and (candidate / "p01.pddl").exists():
        return candidate / "domain.pddl", candidate / "p01.pddl"

    # Otherwise, pick the first folder with domain.pddl and one *.pddl problem.
    for domain_folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        dom = domain_folder / "domain.pddl"
        if not dom.exists():
            continue
        problems = sorted(
            p
            for p in domain_folder.iterdir()
            if p.is_file() and p.suffix == ".pddl" and p.name != "domain.pddl"
        )
        if problems:
            return dom, problems[0]

    raise FileNotFoundError(f"No (domain.pddl, problem.pddl) found under {data_dir}")


def _default_downward_bin(fd_driver: Path) -> Path | None:
    """Best-effort locate the compiled `downward` binary for search-only runs."""
    root = fd_driver.parent
    cand = root / "builds" / "release" / "bin" / "downward"
    if cand.exists():
        return cand
    # Fall back to first match under builds/*/bin/downward
    builds = root / "builds"
    if builds.exists():
        for p in sorted(builds.glob("*/bin/downward")):
            if p.exists():
                return p
    return None


def _sas_cache_path(cache_dir: Path, domain: Path, problem: Path) -> Path:
    """Deterministic per-instance SAS path under cache_dir."""
    # Typical layout: data/<domainname>/domain.pddl + pXX.pddl
    domain_group = domain.parent.name
    return cache_dir / domain_group / f"{problem.stem}.sas"


def main() -> int:
    t0_all = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Smoke test for PDDL syntactic + FD translation feature extraction"
    )
    parser.add_argument(
        "--domain",
        type=Path,
        default=None,
        help="Path to domain.pddl (default: auto-pick from ./data)",
    )
    parser.add_argument(
        "--problem",
        type=Path,
        default=None,
        help="Path to problem .pddl (default: auto-pick from ./data)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Data directory used for auto-pick (default: ./data)",
    )
    parser.add_argument(
        "--fd",
        type=Path,
        default=None,
        help="Path to fast-downward.py (optional). If omitted or missing, FD step is skipped.",
    )

    parser.add_argument(
        "--fd-search-bin",
        type=Path,
        default=None,
        help="Path to compiled `downward` binary (optional). If set, probes can run search-only on cached SAS.",
    )

    parser.add_argument(
        "--sas-cache-dir",
        type=Path,
        default=REPO_ROOT / "sas_cache",
        help="Directory for per-instance SAS cache (default: ./sas_cache)",
    )

    parser.add_argument(
        "--sas-out",
        type=Path,
        default=None,
        help="(Legacy/override) Explicit SAS output path. If set, cache is bypassed for main SAS features.",
    )
    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="Force re-running FD translation even if --sas-out already exists.",
    )
    parser.add_argument(
        "--probe-heuristics-timeout",
        type=float,
        default=3.0,
        help="Timeout seconds for heuristic probe (default: 3.0)",
    )
    parser.add_argument(
        "--probe-landmarks-timeout",
        type=float,
        default=5.0,
        help="Timeout seconds for landmarks probe (default: 5.0)",
    )

    args = parser.parse_args()

    if args.domain is None or args.problem is None:
        domain, problem = _pick_default_instance(args.data_dir)
    else:
        domain, problem = args.domain, args.problem

    print(f"Domain:  {domain}")
    print(f"Problem: {problem}")

    pddl_feats = extract_pddl_syntactic(domain, problem)
    print("\nPDDL syntactic features:")
    print(pddl_feats)

    sas_plus_extractor = _try_load_sas_plus_extractor()

    probe_log_dir = REPO_ROOT / "probe_logs"
    heuristic_log_path = probe_log_dir / "heuristic_relaxed_probe.txt"
    landmark_log_path = probe_log_dir / "landmarks_probe.txt"
    heuristic_sas_path = probe_log_dir / "heuristic_probe.sas"
    landmark_sas_path = probe_log_dir / "landmarks_probe.sas"

    fd_driver = args.fd
    if fd_driver is None:
        # common local names
        for cand in [REPO_ROOT / "fast-downward.py", Path("./fast-downward.py")]:
            if cand.exists():
                fd_driver = cand
                break

    if fd_driver is None or not fd_driver.exists():
        print("\nFD translation features: SKIPPED (fast-downward.py not found)")
        if sas_plus_extractor is not None and args.sas_out.exists():
            print("\nSAS+ structure features:")
            print(sas_plus_extractor(args.sas_out))
        else:
            print("\nSAS+ structure features: SKIPPED (SAS+ extractor or sas_out missing)")
        print("\nHeuristic probe features: SKIPPED (fast-downward.py not found)")
        print("\nLandmark probe features: SKIPPED (fast-downward.py not found)")
        print(f"\nTotal smoke_test elapsed_s={time.perf_counter() - t0_all:.3f}")
        return 0

    fd_search_bin = args.fd_search_bin
    if fd_search_bin is None:
        fd_search_bin = _default_downward_bin(fd_driver)

    # Main SAS path: prefer cache by default, allow explicit override.
    if args.sas_out is not None:
        main_sas_path = args.sas_out
    else:
        main_sas_path = _sas_cache_path(args.sas_cache_dir, domain, problem)
        main_sas_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse existing SAS by default to avoid re-running translation.
    if main_sas_path.exists() and not args.force_translate:
        print(f"\nFD translation features: REUSED (parsing existing {main_sas_path})")
        fd_feats = parse_sas_file(main_sas_path)
        fd_feats.translate_seconds = 0.0
        print(fd_feats)

        if sas_plus_extractor is not None:
            print("\nSAS+ structure features:")
            print(sas_plus_extractor(main_sas_path))
        else:
            print("\nSAS+ structure features: SKIPPED (SAS+ extractor missing)")

        # Probes: always attempt (will run Fast Downward again).
        if extract_heuristic_probe_features is None:
            print("\nHeuristic probe features: SKIPPED (module not available)")
        else:
            print("\nHeuristic probe features:")
            print(
                extract_heuristic_probe_features(
                    fd_driver=fd_driver,
                    fd_search_bin=fd_search_bin,
                    domain=domain,
                    problem=problem,
                    timeout_s=float(args.probe_heuristics_timeout),
                    sas_in=main_sas_path,
                    log_path=heuristic_log_path,
                )
            )

        if extract_landmark_probe_features is None:
            print("\nLandmark probe features: SKIPPED (module not available)")
        else:
            print("\nLandmark probe features:")
            print(
                extract_landmark_probe_features(
                    fd_driver=fd_driver,
                    fd_search_bin=fd_search_bin,
                    domain=domain,
                    problem=problem,
                    timeout_s=float(args.probe_landmarks_timeout),
                    sas_in=main_sas_path,
                    log_path=landmark_log_path,
                )
            )
        print(f"\nTotal smoke_test elapsed_s={time.perf_counter() - t0_all:.3f}")
        return 0

    try:
        fd_feats = extract_fd_translation_features(fd_driver, domain, problem, main_sas_path)
    except Exception as e:
        print(f"\nFD translation features: FAILED ({type(e).__name__}: {e})")
        return 2

    print("\nFD translation features:")
    print(fd_feats)

    if sas_plus_extractor is not None and main_sas_path.exists():
        print("\nSAS+ structure features:")
        print(sas_plus_extractor(main_sas_path))
    else:
        print("\nSAS+ structure features: SKIPPED (SAS+ extractor or sas_out missing)")

    if extract_heuristic_probe_features is None:
        print("\nHeuristic probe features: SKIPPED (module not available)")
    else:
        print("\nHeuristic probe features:")
        print(
            extract_heuristic_probe_features(
                fd_driver=fd_driver,
                fd_search_bin=fd_search_bin,
                domain=domain,
                problem=problem,
                timeout_s=float(args.probe_heuristics_timeout),
                sas_in=main_sas_path,
                log_path=heuristic_log_path,
            )
        )

    if extract_landmark_probe_features is None:
        print("\nLandmark probe features: SKIPPED (module not available)")
    else:
        print("\nLandmark probe features:")
        print(
            extract_landmark_probe_features(
                fd_driver=fd_driver,
                fd_search_bin=fd_search_bin,
                domain=domain,
                problem=problem,
                timeout_s=float(args.probe_landmarks_timeout),
                sas_in=main_sas_path,
                log_path=landmark_log_path,
            )
        )
    print(f"\nTotal smoke_test elapsed_s={time.perf_counter() - t0_all:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
