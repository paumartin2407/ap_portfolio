#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Allow running this file directly without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.PDDL_syn_features import extract_pddl_syntactic  # noqa: E402
from feature_extraction.FD_ins_trans_features import (  # noqa: E402
    extract_fd_translation_features,
)

from feature_extraction.fd_limits import FDMemoryLimitExceeded

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


def _default_downward_bin(fd_driver: Path) -> Optional[Path]:
    """Best-effort locate the compiled `downward` binary for search-only runs."""
    root = fd_driver.parent
    cand = root / "builds" / "release" / "bin" / "downward"
    if cand.exists():
        return cand
    builds = root / "builds"
    if builds.exists():
        for p in sorted(builds.glob("*/bin/downward")):
            if p.exists():
                return p
    return None


def _iter_instances(data_dir: Path) -> Iterable[Tuple[str, Path, Path]]:
    """Yields (domain_dir_name, domain_pddl, problem_pddl)."""
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
            yield domain_folder.name, dom, prob


def _jsonify_value(v: Any) -> Any:
    if isinstance(v, (str, int, float)) or v is None:
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (dict, list, tuple)):
        try:
            return json.dumps(v, sort_keys=True)
        except Exception:
            return str(v)
    return str(v)


def _as_flat_row(prefix: str, obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
    elif isinstance(obj, dict):
        d = obj
    else:
        raise TypeError(f"Unsupported feature object type: {type(obj).__name__}")

    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[f"{prefix}{k}"] = _jsonify_value(v)
    return out


def _read_done_keys(csv_path: Path) -> Set[Tuple[str, str]]:
    """Return {(domain, name)} already present in csv."""
    done: Set[Tuple[str, str]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return done
        if "domain" not in reader.fieldnames or "name" not in reader.fieldnames:
            return done
        for row in reader:
            d = (row.get("domain") or "").strip()
            n = (row.get("name") or "").strip()
            if d and n:
                done.add((d, n))
    return done


def _ensure_header(csv_path: Path, header: List[str]) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        existing = next(r, None)
    if existing is None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
        return

    if existing != header:
        raise RuntimeError(
            "Output CSV header does not match expected header. "
            "Use a fresh output path or keep the feature set constant."
        )


def _append_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, "") for k in header})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract all features for all PDDL problems under data/ and write to a resumable CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing domain folders (default: ./data)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path (rows appended incrementally)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip instances already present in --out-csv (by domain+name)",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=None,
        help="Optional path to append failures as TSV: domain<TAB>name<TAB>error",
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
        help="Path to compiled downward binary (optional; enables search-only probes on temp SAS)",
    )
    parser.add_argument(
        "--probe-heuristics-timeout",
        type=float,
        default=5.0,
        help="Timeout seconds for heuristic probe per instance (default: 5)",
    )
    parser.add_argument(
        "--probe-landmarks-timeout",
        type=float,
        default=5.0,
        help="Timeout seconds for landmarks probe per instance (default: 5)",
    )
    parser.add_argument(
        "--fd-mem-limit-mb",
        type=int,
        default=8192,
        help="Memory cap (MB) applied to each Fast Downward subprocess (default: 8192). Use 0 to disable.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed instances (default: 10; 0 disables)",
    )
    parser.add_argument(
        "--start-after",
        type=str,
        default=None,
        help="Optional: start after a specific key 'domain/name' (useful for manual resume)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process at most N instances",
    )
    args = parser.parse_args()

    fd_driver: Path = args.fd
    if not fd_driver.exists():
        raise FileNotFoundError(f"fast-downward.py not found: {fd_driver}")

    fd_search_bin = args.fd_search_bin
    if fd_search_bin is None:
        fd_search_bin = _default_downward_bin(fd_driver)

    sas_plus_extractor = _try_load_sas_plus_extractor()
    if sas_plus_extractor is None:
        raise RuntimeError("SAS+ extractor missing: feature_extraction/SAS+_struct_features.py")

    if extract_heuristic_probe_features is None:
        raise RuntimeError("Heuristic probe module missing: feature_extraction/heuristic_relaxed_features.py")

    if extract_landmark_probe_features is None:
        raise RuntimeError("Landmark probe module missing: feature_extraction/landmarks_features.py")

    # Build stable header from dataclass field names.
    header: List[str] = ["name", "domain"]

    # PDDL syntactic
    try:
        from feature_extraction.PDDL_syn_features import PDDLSyntacticFeatures  # type: ignore

        for f in PDDLSyntacticFeatures.__dataclass_fields__.keys():  # type: ignore
            header.append(f"pddl_{f}")
    except Exception:
        pass

    # FD translation
    try:
        from feature_extraction.FD_ins_trans_features import FDTranslationFeatures  # type: ignore

        for f in FDTranslationFeatures.__dataclass_fields__.keys():  # type: ignore
            header.append(f"fd_{f}")
    except Exception:
        pass

    # SAS+
    try:
        # Loaded dynamically, but the returned object is a function and we only
        # need the dataclass type for stable headers.
        spec_path = REPO_ROOT / "feature_extraction" / "SAS+_struct_features.py"
        spec = importlib.util.spec_from_file_location("sas_plus_struct_features_types", spec_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            SASStructureFeatures = getattr(module, "SASStructureFeatures", None)
            if SASStructureFeatures is not None and hasattr(SASStructureFeatures, "__dataclass_fields__"):
                for f in SASStructureFeatures.__dataclass_fields__.keys():
                    header.append(f"sas_{f}")
    except Exception:
        pass

    # Heuristic probe
    try:
        from feature_extraction.heuristic_relaxed_features import HeuristicProbeFeatures  # type: ignore

        for f in HeuristicProbeFeatures.__dataclass_fields__.keys():  # type: ignore
            header.append(f"heur_{f}")
    except Exception:
        pass

    # Landmark probe
    try:
        from feature_extraction.landmarks_features import LandmarkProbeFeatures  # type: ignore

        for f in LandmarkProbeFeatures.__dataclass_fields__.keys():  # type: ignore
            header.append(f"lm_{f}")
    except Exception:
        pass

    _ensure_header(args.out_csv, header)

    done_keys: Set[Tuple[str, str]] = _read_done_keys(args.out_csv) if args.resume else set()

    start_after = args.start_after
    start_after_key: Optional[Tuple[str, str]] = None
    if start_after:
        if "/" not in start_after:
            raise ValueError("--start-after must be like 'domain/name'")
        d, n = start_after.split("/", 1)
        start_after_key = (d, n)

    processed = 0
    skipped = 0
    failed = 0
    mem_skipped = 0

    def log_fail(domain_name: str, problem_name: str, msg: str) -> None:
        nonlocal failed
        failed += 1
        if args.failed_log is None:
            return
        args.failed_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_log.open("a", encoding="utf-8") as f:
            f.write(f"{domain_name}\t{problem_name}\t{msg.replace(os.linesep, ' ')}\n")

    t_all = time.perf_counter()

    mem_limit_mb = int(args.fd_mem_limit_mb)
    if mem_limit_mb <= 0:
        mem_limit_mb = 0

    # Reuse one temp dir for the whole run; per-instance SAS file is deleted after row write.
    with tempfile.TemporaryDirectory(prefix="ap_portfolio_tmp_") as td:
        td_path = Path(td)

        for domain_name, dom_pddl, prob_pddl in _iter_instances(args.data_dir):
            problem_name = prob_pddl.name
            key = (domain_name, problem_name)
            if start_after_key is not None:
                # Skip until strictly after the marker.
                if key <= start_after_key:
                    skipped += 1
                    continue

            if key in done_keys:
                skipped += 1
                continue

            if args.limit is not None and processed >= int(args.limit):
                break

            # One temp SAS for the whole instance, deleted after we write the row.
            try:
                # Keep filename short-ish and filesystem-safe.
                sas_path = td_path / f"{domain_name}__{prob_pddl.stem}.sas"

                pddl_feats = extract_pddl_syntactic(dom_pddl, prob_pddl)

                # Translate once.
                fd_feats = extract_fd_translation_features(
                    fd_driver,
                    dom_pddl,
                    prob_pddl,
                    sas_path,
                    mem_limit_mb=(mem_limit_mb or None),
                )

                # Structural SAS+.
                sas_feats = sas_plus_extractor(sas_path)

                # Probes (search-only on SAS stdin when possible).
                heur_feats = extract_heuristic_probe_features(
                    fd_driver=fd_driver,
                    fd_search_bin=fd_search_bin,
                    domain=dom_pddl,
                    problem=prob_pddl,
                    timeout_s=float(args.probe_heuristics_timeout),
                    sas_in=sas_path,
                    log_path=None,
                    mem_limit_mb=(mem_limit_mb or None),
                )

                lm_feats = extract_landmark_probe_features(
                    fd_driver=fd_driver,
                    fd_search_bin=fd_search_bin,
                    domain=dom_pddl,
                    problem=prob_pddl,
                    timeout_s=float(args.probe_landmarks_timeout),
                    sas_in=sas_path,
                    sas_out=None,
                    log_path=None,
                    mem_limit_mb=(mem_limit_mb or None),
                )

                row: Dict[str, Any] = {"name": problem_name, "domain": domain_name}
                row.update(_as_flat_row("pddl_", pddl_feats))
                row.update(_as_flat_row("fd_", fd_feats))
                row.update(_as_flat_row("sas_", sas_feats))
                row.update(_as_flat_row("heur_", heur_feats))
                row.update(_as_flat_row("lm_", lm_feats))

                _append_row(args.out_csv, header, row)

                try:
                    sas_path.unlink(missing_ok=True)
                except Exception:
                    pass

                processed += 1

                pe = int(args.progress_every)
                if pe > 0 and processed % pe == 0:
                    elapsed = time.perf_counter() - t_all
                    avg = elapsed / processed
                    remaining = None
                    if args.limit is not None:
                        remaining = max(0, int(args.limit) - processed)
                    if remaining is not None:
                        eta = remaining * avg
                        print(
                            f"progress processed={processed} skipped={skipped} failed={failed} "
                            f"avg_s={avg:.1f} eta_s={eta:.0f}"
                        )
                    else:
                        print(
                            f"progress processed={processed} skipped={skipped} failed={failed} "
                            f"avg_s={avg:.1f}"
                        )
            except FDMemoryLimitExceeded:
                mem_skipped += 1
                if args.failed_log is not None:
                    log_fail(domain_name, problem_name, "FDMemoryLimitExceeded")
                continue
            except Exception as e:
                log_fail(domain_name, problem_name, f"{type(e).__name__}: {e}")
                continue

    elapsed_all = time.perf_counter() - t_all
    print(
        f"Done. processed={processed} skipped={skipped} failed={failed} mem_skipped={mem_skipped} "
        f"elapsed_s={elapsed_all:.1f} out={args.out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
