# Label generation (Fast Downward portfolio)

This directory contains a small runner to execute the 3-solver portfolio described in `solvers.md` over the PDDL instances under `data/` and produce **labels** for ML.

Each output row is:

- `name`: problem filename (e.g. `p01.pddl`)
- `domain`: domain folder name (e.g. `agricola-sat18-strips`)
- `solved`: `1` if a plan was found within limits, else `0`
- `time_seconds`: Fast Downward reported `INFO     Planner time: ...s` **only if** solved; otherwise `NaN`

The output CSV is intended to be merged later into `all_features.csv` using the key `(name, domain)`.

## Assumptions

- Your Fast Downward entrypoint is a `fast-downward.py` script.
- Each domain folder in `data/<domain>/` contains `domain.pddl` plus one or more `*.pddl` problem files.

## Usage

From repo root:

```bash
python3 -m label_generation.run_fd_labels \
  --solver 1 \
  --data-dir data \
  --fd-path /home/pauma/downward/fast-downward.py \
  --out solvers_labels/solver1.csv
```

### Limits

Defaults match your request:

- time limit: 1800 seconds (30 minutes)
- memory cap: 8 GB

The runner enforces limits in two ways:

- passes Fast Downward driver options: `--overall-time-limit` (seconds) and `--overall-memory-limit` (MiB)
- additionally uses an external wall-clock timeout as a safety net

### Resume

The runner is resume-safe by default:

- if `--out` exists, it loads all `(name, domain)` already present and **skips** them.
- it appends one row per completed run and flushes immediately.

Overwrite:

```bash
python3 -m label_generation.run_fd_labels \
  --solver 1 \
  --data-dir data \
  --fd-path /home/pauma/downward/fast-downward.py \
  --out solvers_labels/solver1.csv \
  --overwrite
```

## Fixing an existing CSV (re-parse times)

If you generated a CSV before switching to parsing `Planner time`, you can rewrite the times from the stored logs:

```bash
python3 -m label_generation.fix_times_from_logs \
  --solver 1 \
  --csv solvers_labels/smoke_solver1.csv
```

## Outputs

- CSV labels: whatever you pass via `--out`
- Logs: `label_generation/logs/solver_<id>/<domain>/<problem>.log`
