"""Execute a resumable visual-corruption plan and save episode-level results."""

import csv
import fcntl
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser

from corruptions import make_corruption
from runner import load_policy_components, make_task_environment, run_batch


FIELDNAMES = [
    "condition",
    "method",
    "camera",
    "kernel_size",
    "freeze_period",
    "parameters",
    "suite",
    "task_id",
    "seed",
    "success",
    "steps",
    "total_reward",
    "max_reward",
]


@dataclass
class ExperimentConfig:
    plan_path: Path = Path("configs/visual_robustness_pilot.json")
    output_csv: Path = Path("results/visual_robustness_pilot.csv")
    policy_path: str = "HuggingFaceVLA/smolvla_libero"
    save_videos: bool = False
    resume: bool = True
    rerun_conditions: str = ""


def load_plan(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)

    required_keys = {"suite", "task_ids", "seeds", "conditions"}
    missing = required_keys - plan.keys()
    if missing:
        raise ValueError(f"Plan is missing keys: {', '.join(sorted(missing))}")
    if not plan["seeds"]:
        raise ValueError("Plan must contain at least one seed")
    return plan


def _condition_name(condition: dict) -> str:
    return condition.get("name", condition["method"])


def _camera_label(condition: dict) -> str:
    if condition.get("schedule"):
        return ",".join(dict.fromkeys(item["camera"] for item in condition["schedule"]))
    if condition.get("periods"):
        return ",".join(condition["periods"])
    camera = condition.get("cameras", condition.get("camera", "image2"))
    return ",".join(camera) if isinstance(camera, list) else camera


def _make_condition_corruption(condition: dict):
    return make_corruption(
        condition["method"],
        kernel_size=condition.get("kernel_size", 5),
        camera=condition.get("cameras", condition.get("camera", "image2")),
        period=condition.get("period", 1),
        start_step=condition.get("start_step", 0),
        duration=condition.get("duration", 1),
        schedule=condition.get("schedule"),
        periods=condition.get("periods"),
        fraction=condition.get("fraction", 0.25),
        name=_condition_name(condition),
    )


def _load_existing_rows(path: Path, resume: bool) -> list[dict]:
    if not resume or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                f"Cannot resume {path}: CSV columns do not match the current experiment format"
            )
        rows = list(reader)

    keys = [(row["condition"], int(row["task_id"]), int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Cannot resume {path}: duplicate condition/task/seed rows found")
    return rows


def _success_as_bool(value) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def _write_summary(rows: list[dict], path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["task_id"]))].append(_success_as_bool(row["success"]))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["condition", "task_id", "n_episodes", "successes", "success_rate"],
        )
        writer.writeheader()
        for (condition_name, task_id), successes in sorted(grouped.items()):
            writer.writerow(
                {
                    "condition": condition_name,
                    "task_id": task_id,
                    "n_episodes": len(successes),
                    "successes": sum(successes),
                    "success_rate": sum(successes) / len(successes),
                }
            )


@parser.wrap()
def main(cfg: ExperimentConfig):
    plan = load_plan(cfg.plan_path)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.output_csv.with_suffix(f"{cfg.output_csv.suffix}.lock")
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(f"Experiment is already writing to {cfg.output_csv}") from error
    summary_csv = cfg.output_csv.with_name(f"{cfg.output_csv.stem}_summary.csv")

    rows = _load_existing_rows(cfg.output_csv, cfg.resume)
    rerun_conditions = {
        name.strip() for name in cfg.rerun_conditions.split(",") if name.strip()
    }
    plan_conditions = {_condition_name(condition) for condition in plan["conditions"]}
    unknown_reruns = rerun_conditions - plan_conditions
    if unknown_reruns:
        raise ValueError(f"Unknown rerun condition(s): {sorted(unknown_reruns)}")
    if rerun_conditions:
        rows = [row for row in rows if row["condition"] not in rerun_conditions]
        with cfg.output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Removed previous rows for rerun: {sorted(rerun_conditions)}")
    completed = {
        (row["condition"], int(row["task_id"]), int(row["seed"])) for row in rows
    }
    expected = {
        (_condition_name(condition), int(task_id), int(seed))
        for condition in plan["conditions"]
        for task_id in plan["task_ids"]
        for seed in plan["seeds"]
    }
    unexpected = completed - expected
    if unexpected:
        raise ValueError(f"Output contains rows not present in this plan: {sorted(unexpected)[:3]}")
    if completed == expected:
        _write_summary(rows, summary_csv)
        print(f"Experiment already complete: {cfg.output_csv} ({len(rows)} rows)")
        return

    n_envs = plan.get("n_envs", len(plan["seeds"]))
    if n_envs != len(plan["seeds"]):
        raise ValueError("n_envs must equal the number of seeds in the plan")

    attention_config = plan.get("attention", {})
    attention_enabled = attention_config.get("enabled", False)
    attention_csv = attention_config.get("output_csv")
    if attention_enabled and attention_csv is None:
        raise ValueError("attention.enabled requires attention.output_csv")

    policy, policy_config, preprocessor, postprocessor = load_policy_components(cfg.policy_path)
    append = bool(rows) and cfg.resume
    mode = "a" if append else "w"
    with cfg.output_csv.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not append:
            writer.writeheader()

        for condition in plan["conditions"]:
            method = condition["method"]
            condition_name = _condition_name(condition)
            camera_label = _camera_label(condition)
            corruption = _make_condition_corruption(condition)
            parameters = {
                key: condition[key]
                for key in ("kernel_size", "period", "start_step", "duration", "schedule", "periods", "fraction")
                if key in condition
            }
            print(f"\nCondition: {condition_name}")

            for task_id in plan["task_ids"]:
                task_keys = {(condition_name, int(task_id), int(seed)) for seed in plan["seeds"]}
                if task_keys <= completed:
                    print(f"Skipping completed task_id={task_id}")
                    continue

                env, env_preprocessor, env_postprocessor = make_task_environment(
                    plan["suite"], task_id, policy_config, n_envs=n_envs
                )
                try:
                    output_names = None
                    videos_dir = None
                    if cfg.save_videos:
                        output_names = [
                            f"{condition_name}_task{task_id}_seed{seed}" for seed in plan["seeds"]
                        ]
                        videos_dir = (
                            cfg.output_csv.parent
                            / "videos"
                            / cfg.output_csv.stem
                            / condition_name
                        )

                    attention_metadata = [
                        {
                            "condition": condition_name,
                            "method": method,
                            "corrupted_camera": camera_label if method != "clean" else "none",
                            "kernel_size": condition.get("kernel_size", ""),
                            "parameters": parameters,
                            "suite": plan["suite"],
                            "task_id": task_id,
                            "seed": seed,
                        }
                        for seed in plan["seeds"]
                    ]

                    results = run_batch(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seeds=plan["seeds"],
                        max_steps=plan.get("max_steps"),
                        corruption=corruption,
                        output_names=output_names,
                        videos_dir=videos_dir,
                        attention=attention_enabled,
                        attention_csv=attention_csv,
                        attention_metadata=attention_metadata,
                        attention_layers=attention_config.get("layers"),
                        attention_steps=attention_config.get("steps"),
                    )
                    for result in results:
                        key = (condition_name, int(task_id), int(result["seed"]))
                        if key in completed:
                            continue
                        row = {
                            "condition": condition_name,
                            "method": method,
                            "camera": camera_label if method != "clean" else "none",
                            "kernel_size": condition.get("kernel_size", ""),
                            "freeze_period": condition.get("period", ""),
                            "parameters": json.dumps(parameters, separators=(",", ":")),
                            "suite": plan["suite"],
                            "task_id": task_id,
                            **result,
                        }
                        writer.writerow(row)
                        file.flush()
                        rows.append(row)
                        completed.add(key)
                        print(row)
                    _write_summary(rows, summary_csv)
                finally:
                    env.close()

    missing = expected - completed
    if missing:
        raise RuntimeError(f"Experiment ended with {len(missing)} missing rows")
    _write_summary(rows, summary_csv)
    print(f"\nPer-episode results: {cfg.output_csv}")
    print(f"Success-rate summary: {summary_csv}")


if __name__ == "__main__":
    main()
