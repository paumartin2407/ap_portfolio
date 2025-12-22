from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess
import time
import re

from feature_extraction.fd_limits import (
	FDMemoryLimitExceeded,
	looks_like_memory_limit,
	preexec_set_memory_limit,
)


def _run_capture(cmd: List[str], timeout_s: float, mem_limit_mb: Optional[int] = None) -> Tuple[str, float, bool, int]:
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


def _run_capture_sas_stdin(cmd: List[str], sas_in: Path, timeout_s: float, mem_limit_mb: Optional[int] = None) -> Tuple[str, float, bool, int]:
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


def _all_ints(patterns: List[str], text: str) -> List[int]:
	vals: List[int] = []
	for pat in patterns:
		for m in re.finditer(pat, text, flags=re.IGNORECASE):
			try:
				vals.append(int(m.group(1)))
			except Exception:
				pass
	return vals


def _parse_reasonable_ordering_delta(text: str) -> Tuple[int, int, int]:
	"""Best-effort extraction of (before, after, delta) orderings around the
	'approx. reasonable orders' phase in lm_reasonable_orders_hps output.

	Returns (-1, -1, -1) if the marker or surrounding counts aren't found.
	"""
	lines = text.splitlines()
	marker_idx = -1
	for i, line in enumerate(lines):
		if re.search(r"approx\.\s*reasonable\s+orders", line, flags=re.IGNORECASE):
			marker_idx = i
			break
	if marker_idx < 0:
		return -1, -1, -1

	order_pat = re.compile(r"there\s+are\s+(\d+)\s+landmark\s+orderings", flags=re.IGNORECASE)

	before = -1
	for j in range(marker_idx - 1, -1, -1):
		m = order_pat.search(lines[j])
		if m:
			try:
				before = int(m.group(1))
			except Exception:
				before = -1
			break

	after = -1
	for j in range(marker_idx + 1, len(lines)):
		m = order_pat.search(lines[j])
		if m:
			try:
				after = int(m.group(1))
			except Exception:
				after = -1
			break

	if before < 0 or after < 0:
		return -1, -1, -1

	return before, after, max(0, after - before)


def _parse_goal_count_from_sas(sas_path: Path) -> int:
	# SAS goal section:
	# begin_goal
	# <k>
	# <var val> lines...
	# end_goal
	if not sas_path.exists():
		return -1
	lines = sas_path.read_text(encoding="utf-8", errors="ignore").splitlines()
	for i, line in enumerate(lines):
		if line.strip() == "begin_goal" and i + 1 < len(lines):
			try:
				return int(lines[i + 1].strip())
			except Exception:
				return -1
	return -1


@dataclass
class LandmarkProbeFeatures:
	probe_seconds: float

	lm_num_landmarks: int
	lm_num_orderings_total: int

	lm_num_reasonable_orders: int
	lm_num_necessary_orders: int
	lm_num_greedy_necessary_orders: int

	lm_num_conjunctive_landmarks: int
	lm_num_disjunctive_landmarks: int

	goal_count: int
	landmarks_per_goal: float
	orderings_per_landmark: float


def extract_landmark_probe_features(
	fd_driver: Path,
	domain: Path,
	problem: Path,
	fd_search_bin: Optional[Path] = None,
	timeout_s: float = 5.0,
	sas_in: Optional[Path] = None,
	sas_out: Optional[Path] = None,
	log_path: Optional[Path] = None,
	mem_limit_mb: Optional[int] = None,
) -> LandmarkProbeFeatures:
	"""Runs a short FD probe and parses landmark stats.

	Fast path: if `fd_search_bin` + `sas_in` exist, runs search-only on the cached SAS.
	Fallback: runs `fast-downward.py --alias lama-first` on (domain, problem).
	"""

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
		# Approximate `--alias lama-first` search config (based on your build output).
		lama_first_search = (
			"let(hlm,landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false),"
			"let(hff,ff(transform=adapt_costs(one)),lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,reopen_closed=false)))"
		)
		cmd: List[str] = [str(fd_search_bin)]
		if plan_file is not None:
			cmd += ["--internal-plan-file", plan_file]
		cmd += ["--search", lama_first_search]
		out, elapsed, timed_out, rc = _run_capture_sas_stdin(cmd, Path(sas_in), timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)
	else:
		cmd = [str(fd_driver), "--alias", "lama-first", str(domain), str(problem)]
		if sas_out is not None:
			cmd = [str(fd_driver), "--alias", "lama-first", "--sas-file", str(sas_out), str(domain), str(problem)]
		out, elapsed, timed_out, rc = _run_capture(cmd, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)

	if (not timed_out) and rc != 0 and looks_like_memory_limit(out, rc):
		raise FDMemoryLimitExceeded("FD landmarks probe hit memory limit")

	if log_path is not None:
		try:
			log_path.parent.mkdir(parents=True, exist_ok=True)
			with log_path.open("w", encoding="utf-8") as f:
				f.write("# Landmark probe (Fast Downward)\n")
				f.write(f"timeout_s: {timeout_s}\n")
				f.write(f"timed_out: {int(timed_out)}\n")
				f.write(f"return_code: {rc}\n")
				f.write(f"elapsed_s: {elapsed}\n\n")
				f.write("command:\n")
				f.write(" ".join(map(str, cmd)) + "\n\n")
				if can_search_only and sas_in is not None:
					f.write(f"stdin_sas: {sas_in}\n\n")
				f.write("output:\n")
				f.write(out)
				if out and not out.endswith("\n"):
					f.write("\n")
		except Exception:
			pass

	lm_num_landmarks = _first_int([
		r"discovered\s+(\d+)\s+landmarks",
		r"found\s+(\d+)\s+landmarks",
		r"(\d+)\s+landmarks\s+found",
		r"landmarks\s*:\s*(\d+)",
		r"number\s+of\s+landmarks\s*:\s*(\d+)",
	], out)

	orderings_all = _all_ints([
		r"(\d+)\s+landmark\s+orderings",
		r"there\s+are\s+(\d+)\s+landmark\s+orderings",
		r"orderings\s*:\s*(\d+)",
		r"number\s+of\s+orderings\s*:\s*(\d+)",
	], out)
	lm_num_orderings_total = orderings_all[-1] if orderings_all else -1

	before_ord, _after_ord, delta_reasonable = _parse_reasonable_ordering_delta(out)
	lm_num_reasonable = delta_reasonable if delta_reasonable >= 0 else -1
	lm_num_greedy_necessary = before_ord if before_ord >= 0 else -1
	lm_num_necessary = 0 if before_ord >= 0 else -1

	lm_num_conj = _first_int([
		r"conjunctive\s+landmarks\s*:\s*(\d+)",
		r"conjunctive\s*:\s*(\d+)",
		r"disjunctive\s+and\s+(\d+)\s+are\s+conjunctive",
	], out)

	lm_num_disj = _first_int([
		r"disjunctive\s+landmarks\s*:\s*(\d+)",
		r"disjunctive\s*:\s*(\d+)",
		r"of\s+which\s+(\d+)\s+are\s+disjunctive",
	], out)

	sas_for_stats = sas_in if (sas_in is not None and Path(sas_in).exists()) else sas_out
	goal_count = _parse_goal_count_from_sas(Path(sas_for_stats)) if sas_for_stats is not None else -1

	landmarks_per_goal = float(lm_num_landmarks) / goal_count if (lm_num_landmarks >= 0 and goal_count and goal_count > 0) else 0.0
	orderings_per_landmark = float(lm_num_orderings_total) / lm_num_landmarks if (lm_num_orderings_total >= 0 and lm_num_landmarks and lm_num_landmarks > 0) else 0.0

	return LandmarkProbeFeatures(
		probe_seconds=float(elapsed),
		lm_num_landmarks=int(lm_num_landmarks),
		lm_num_orderings_total=int(lm_num_orderings_total),
		lm_num_reasonable_orders=int(lm_num_reasonable),
		lm_num_necessary_orders=int(lm_num_necessary),
		lm_num_greedy_necessary_orders=int(lm_num_greedy_necessary),
		lm_num_conjunctive_landmarks=int(lm_num_conj),
		lm_num_disjunctive_landmarks=int(lm_num_disj),
		goal_count=int(goal_count),
		landmarks_per_goal=float(landmarks_per_goal),
		orderings_per_landmark=float(orderings_per_landmark),
	)
