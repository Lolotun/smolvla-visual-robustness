# SmolVLA visual robustness: experiment log

Topic: **When Vision Freezes: Robustness of SmolVLA to Camera Failures in LIBERO**.

This file records what has been tested, what results were obtained, and what to run next.
The combined analysis of all finished runs is in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## Common protocol

- Policy: `HuggingFaceVLA/smolvla_libero`, no fine-tuning.
- Suite: `libero_spatial`.
- Tasks: `task_id = 0, 1, 2, 3, 4`.
- Seeds: `42, 43, 44`.
- Maximum episode length: 140 policy steps.
- Policy and observation rate: 20 Hz.
- Per condition: 5 tasks × 3 seeds = 15 episodes.
- `image` / suffix `_0`: main camera.
- `image2` / suffix `_1`: wrist camera.

## Completed

### Gaussian blur

Status: **done** for both cameras and kernels `5, 11, 15, 31, 51`.

Results: [`results/visual_robustness_5tasks_3seeds_batch.csv`](results/visual_robustness_5tasks_3seeds_batch.csv).

| Condition | Success rate |
|---|---:|
| clean | 73.3% (11/15) |
| blur 5, main | 40.0% (6/15) |
| blur 11, main | 73.3% (11/15) |
| blur 15, main | 33.3% (5/15) |
| blur 31, main | 20.0% (3/15) |
| blur 51, main | 20.0% (3/15) |
| blur 5, wrist | 73.3% (11/15) |
| blur 11, wrist | 53.3% (8/15) |
| blur 15, wrist | 46.7% (7/15) |
| blur 31, wrist | 53.3% (8/15) |
| blur 51, wrist | 13.3% (2/15) |

Preliminary conclusion: strong blur degrades both cameras, but 15 episodes per condition are not enough to interpret
non-monotonic values as a real effect. The same grid was repeated in the attention run
([`results/attention/rollouts_5tasks_3seeds.csv`](results/attention/rollouts_5tasks_3seeds.csv)); the analysis
notebook pools both runs (30 episodes per condition).

### Camera off for the whole episode

Status: **done**.

| Condition | Success rate |
|---|---:|
| turnoff main | 13.3% (2/15) |
| turnoff wrist | 0.0% (0/15) |

Results are in the same CSV as blur.

### Lower camera rate via frame freeze

Status: **done** for both cameras, periods `2, 5, 10, 20`.

`freeze_P_C` refreshes camera `C` every `P` policy steps and holds the previous frame in between.

Main camera results: [`results/freeze_image0_5tasks_3seeds.csv`](results/freeze_image0_5tasks_3seeds.csv).

| Condition | Success rate | Change vs clean of this run |
|---|---:|---:|
| clean | 66.7% (10/15) | — |
| freeze 2, main | 73.3% (11/15) | +6.6 pp |
| freeze 5, main | 53.3% (8/15) | −13.4 pp |
| freeze 10, main | 46.7% (7/15) | −20.0 pp |
| freeze 20, main | 6.7% (1/15) | −60.0 pp |

Wrist camera results: [`results/freeze_image1_5tasks_3seeds.csv`](results/freeze_image1_5tasks_3seeds.csv).

| Condition | Success rate | Change vs clean of this run |
|---|---:|---:|
| clean | 73.3% (11/15) | — |
| freeze 2, wrist | 53.3% (8/15) | −20.0 pp |
| freeze 5, wrist | 53.3% (8/15) | −20.0 pp |
| freeze 10, wrist | 53.3% (8/15) | −20.0 pp |
| freeze 20, wrist | 26.7% (4/15) | −46.6 pp |

Main observation: at period 20 the main camera is more critical — success drops to 6.7%, versus 26.7% for the wrist camera.

Full freeze of the first frame is also done:

- fresh clean: 53.3% (8/15);
- `freeze_140_0`: 0.0% (0/15), the first real main-camera frame is held for the whole episode;
- `freeze_140_1`: 0.0% (0/15), the first real wrist-camera frame is held for the whole episode.

Period 140 equals the maximum episode length. The rollout runs steps `0–139`, so the camera is never refreshed again.

Results: [`results/freeze_full_5tasks_3seeds.csv`](results/freeze_full_5tasks_3seeds.csv). Checked: 45 unique rows and 90 videos at 20 FPS.

### Cross-attention pilot

Status: **pilot done** for `libero_spatial`, `task_id=0`, `seed=42`, clean and different blur levels on both cameras.

- Attention CSV: [`results/attention/libero_spatial_task0_seed42.csv`](results/attention/libero_spatial_task0_seed42.csv).
- Rollout CSV: [`results/attention/rollouts_task0_seed42.csv`](results/attention/rollouts_task0_seed42.csv).
- Analysis: [`notebooks/effects.ipynb`](notebooks/effects.ipynb).

The full attention run (5 tasks × 3 seeds, blur and turn-off on both cameras, attention at step 0) is in
[`results/attention/libero_spatial_5tasks_3seeds.csv`](results/attention/libero_spatial_5tasks_3seeds.csv); the
controlled probe on identical clean observations is in [`results/controlled_attention/`](results/controlled_attention).
Both are analysed in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

### One temporary blackout per episode

Status: **done and checked**.

In each corrupted episode only one camera is switched off once for 20 steps. For each camera the start, middle and
later part of the episode are tested separately.

| Condition | Camera | Steps | Approximate phase |
|---|---|---:|---|
| `off_approach_0` | main | 10–29 | moving towards the object |
| `off_grasp_0` | main | 40–59 | approach and grasp |
| `off_transport_0` | main | 60–79 | carrying to the target |
| `off_approach_1` | wrist | 10–29 | moving towards the object |
| `off_grasp_1` | wrist | 40–59 | approach and grasp |
| `off_transport_1` | wrist | 60–79 | carrying to the target |

Condition parameters:

```json
[
  {"name": "off_approach_0", "method": "temporary_off", "camera": "image", "start_step": 10, "duration": 20},
  {"name": "off_grasp_0", "method": "temporary_off", "camera": "image", "start_step": 40, "duration": 20},
  {"name": "off_transport_0", "method": "temporary_off", "camera": "image", "start_step": 60, "duration": 20},
  {"name": "off_approach_1", "method": "temporary_off", "camera": "image2", "start_step": 10, "duration": 20},
  {"name": "off_grasp_1", "method": "temporary_off", "camera": "image2", "start_step": 40, "duration": 20},
  {"name": "off_transport_1", "method": "temporary_off", "camera": "image2", "start_step": 60, "duration": 20}
]
```

Size including a fresh clean baseline: 7 conditions × 5 tasks × 3 seeds = 105 episodes.

Results: [`results/temporary_off_single_5tasks_3seeds.csv`](results/temporary_off_single_5tasks_3seeds.csv).

| Condition | Success rate | Change vs fresh clean |
|---|---:|---:|
| clean | 86.7% (13/15) | — |
| `off_approach_0` | 66.7% (10/15) | −20.0 pp |
| `off_grasp_0` | 53.3% (8/15) | −33.4 pp |
| `off_transport_0` | 53.3% (8/15) | −33.4 pp |
| `off_approach_1` | 46.7% (7/15) | −40.0 pp |
| `off_grasp_1` | 60.0% (9/15) | −26.7 pp |
| `off_transport_1` | 66.7% (10/15) | −20.0 pp |

Preliminary observation: the wrist camera is most sensitive to the early window `10–29` (−40.0 pp). For the main
camera the grasp and transport windows give the same drop: both 53.3% (−33.4 pp). Checked: 105 unique rows and
210 videos at 20 FPS; transport videos and CSV were regenerated for the exact `60–79` window.

### Three temporary blackouts within one episode

Status: **partially run**. `off_mix_0_1_0` finished all 15 episodes; `off_mix_1_0_1` stopped after 3 of 15.

In each corrupted episode there are three 20-step blackouts. The cameras alternate, and the second schedule swaps them.

| Condition | Steps 10–29 | Steps 40–59 | Steps 80–99 |
|---|---|---|---|
| `off_mix_0_1_0` | main off | wrist off | main off |
| `off_mix_1_0_1` | wrist off | main off | wrist off |

Condition parameters: [`configs/temporary_off_mix.json`](configs/temporary_off_mix.json).

Results: [`results/temporary_off_mix_5tasks_3seeds.csv`](results/temporary_off_mix_5tasks_3seeds.csv).

| Condition | Success rate |
|---|---:|
| clean | 53.3% (8/15) |
| `off_mix_0_1_0` | 33.3% (5/15) |
| `off_mix_1_0_1` | 33.3% (1/3), incomplete |

Remaining: finish `off_mix_1_0_1` (12 episodes).

## Possible follow-ups after temporary blackout

- Constant image delay relative to the robot state.
- Action-deviation analysis on identical clean/corrupted observations.

## Planned

### Simultaneous freeze of both cameras at different rates

Status: **config ready** ([`configs/multi_frequency_freeze.json`](configs/multi_frequency_freeze.json)), **not run yet**.

Both cameras are frozen at the same time. Neighbouring rate levels `10↔4 Hz`, `4↔2 Hz` and `2↔1 Hz` are tested;
within each pair the rates are swapped between the main and wrist cameras.

| Condition | Main period | Main rate | Wrist period | Wrist rate |
|---|---:|---:|---:|---:|
| `freeze_main5_wrist2` | 5 | 4 Hz | 2 | 10 Hz |
| `freeze_main2_wrist5` | 2 | 10 Hz | 5 | 4 Hz |
| `freeze_main10_wrist5` | 10 | 2 Hz | 5 | 4 Hz |
| `freeze_main5_wrist10` | 5 | 4 Hz | 10 | 2 Hz |
| `freeze_main20_wrist10` | 20 | 1 Hz | 10 | 2 Hz |
| `freeze_main10_wrist20` | 10 | 2 Hz | 20 | 1 Hz |

Size: 6 conditions × 5 tasks × 3 seeds = 90 episodes (105 with a fresh clean baseline). This grid changes the rate of
both cameras and tests which one needs fresher images more, without the full `2 × 5 × 10 × 20` grid.

### Partial occlusion from the right edge

Status: **config ready** ([`configs/right_occlusion.json`](configs/right_occlusion.json)), **not run yet**.

An opaque black area enters the selected camera from the right edge for the whole episode. It covers a given fraction
of the image width over the full height. The other camera is unchanged.

| Condition | Camera | Covered part of the image |
|---|---|---:|
| `right_occlusion_25_0` | main | right 25% |
| `right_occlusion_50_0` | main | right 50% |
| `right_occlusion_75_0` | main | right 75% |
| `right_occlusion_25_1` | wrist | right 25% |
| `right_occlusion_50_1` | wrist | right 50% |
| `right_occlusion_75_1` | wrist | right 75% |

Size including a fresh clean baseline: 7 conditions × 5 tasks × 3 seeds = 105 episodes.

## Execution procedure

### 1. Reliable runs

- Do not modify the installed LeRobot package inside `.venv`; all logic stays in `src/`.
- Save videos at 20 FPS, matching policy steps.
- Separate videos by experiment name so new clean videos do not overwrite old ones.
- Support safe resumption of an interrupted plan: already saved `(condition, task_id, seed)` are not re-run.
- Write the CSV row right after each episode.
- Run only one GPU experiment at a time.

### 2. Check each corruption before the full run

- Check Python and JSON syntax.
- On synthetic NumPy frames, check exact step boundaries, the selected camera, array shape and `uint8`.
- Do a short smoke run on one task and one seed.
- Check the saved video and make sure the corruption is visible on the right camera.
- Do not include smoke results in the final statistics.

### 3. Run the sets sequentially

Each set uses `libero_spatial`, tasks `0–4`, seeds `42, 43, 44`, `n_envs=3`, at most 140 steps and its own fresh clean baseline.

1. **Full freeze:** `clean`, `freeze_140_0`, `freeze_140_1` — 45 episodes. Done.
2. **One blackout per episode:** `clean` and 6 phase/camera conditions — 105 episodes. Done.
3. **Three blackouts per episode:** `clean`, `off_mix_0_1_0`, `off_mix_1_0_1` — 45 episodes. 33 of 45 done.
4. **Different rates for the two cameras:** `clean` and 6 frequency conditions — 105 episodes.
5. **Right occlusion:** `clean` and 6 conditions `25/50/75%` for two cameras — 105 episodes.

### 4. Checks after each set

- Check the expected number of rows and that all three seeds exist for every task and condition.
- Check that there are no duplicate `(condition, task_id, seed)`.
- Create summaries by condition and by task.
- Compute success rate, drop relative to clean and mean number of steps.
- Check the number and FPS of videos; for temporary events check the frame intervals.
- Record the status and main results in this file before moving to the next set.

### 5. Final analysis

Done in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) for all finished sets:

- single table of all results;
- success-rate plots for each camera and severity;
- main vs wrist comparison at equal corruption;
- per-task breakdown and time-to-success curves;
- cross-attention analysis.

Still open: paired action-deviation analysis on identical clean/corrupted observations. The number of seeds stays
`42, 43, 44` in all final experiments.

## Limitations and technical notes

- One episode out of 15 changes the success rate by 6.7 percentage points. Small differences cannot be treated as reliable effects.
- Clean gave between 53.3% and 86.7% in separate runs. Evaluation is not fully deterministic, so every comparison
  should use the clean episodes of the same run or the pooled clean baseline.
- Older videos were recorded with `render_fps=80`, although observations arrive at 20 Hz. They play about four times
  too fast; watch them at `0.25×` or record new videos at 20 FPS.
- Videos show images after corruption. In `freeze_*_0` the frozen main camera is visible in `*_main.mp4`, in
  `freeze_*_1` the frozen wrist camera is visible in `*_wrist.mp4`.
