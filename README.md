## SmolVLA visual robustness in LIBERO

This repository evaluates the pretrained `HuggingFaceVLA/smolvla_libero` policy
under controlled camera corruptions in LIBERO.

### Repository layout

- `src/runner.py` — one rollout. It loads the policy, steps the environment,
  applies a corruption before preprocessing, and returns episode metrics.
- `src/corruptions.py` — all image corruptions. Add new interventions here.
- `src/experiment.py` — reads a plan from `configs/*.json`, reuses one loaded
  policy, executes every condition/task/seed, and writes per-episode CSV data.
- `src/attention_probe.py` — optional expert-to-image cross-attention logging.
- `configs/` — declarative experiment plans.
- `results/` — rollout CSVs, success-rate summaries, attention CSVs, and videos.
- `notebooks/effects.ipynb` — plots and tables from saved CSV files.

### Run an experiment

```bash
export MUJOCO_GL=egl

UV_CACHE_DIR=/tmp/uv-cache uv run python src/experiment.py \
  --plan_path=configs/freeze.json \
  --output_csv=results/freeze_5tasks_3seeds.csv
```

The command writes one row per `(condition, task_id, seed)` and a second
`*_summary.csv` file with success rates by task and condition.

### Freeze definition

`freeze_period_P_C` refreshes camera `C` only at steps `0, P, 2P, ...`.
At all other control steps, SmolVLA receives a copy of the most recently seen
frame from that camera. For example, `freeze_5_1` updates the wrist camera once
every five control steps and holds its old image for four steps.

`configs/freeze.json` uses the same suite, task IDs, seeds, environment batch
size, and 140-step limit as the blur benchmark, so success rates are comparable.
