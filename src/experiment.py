"""Execute a visual-corruption plan and save per-episode and summary CSV files."""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser

from corruptions import make_corruption
from runner import load_policy_components, make_task_environment, run_batch


@dataclass
class ExperimentConfig:
    plan_path: Path = Path("configs/visual_robustness_pilot.json")
    output_csv: Path = Path("results/visual_robustness_pilot.csv")
    save_videos: bool = False


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


@parser.wrap()
def main(cfg: ExperimentConfig):
    plan = load_plan(cfg.plan_path)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = cfg.output_csv.with_name(f"{cfg.output_csv.stem}_summary.csv")

    policy, policy_config, preprocessor, postprocessor = load_policy_components()
    fieldnames = [
        "condition",
        "method",
        "camera",
        "kernel_size",
        "suite",
        "task_id",
        "seed",
        "success",
        "steps",
        "total_reward",
        "max_reward",
    ]
    grouped_successes = defaultdict(list)
    n_envs = plan.get("n_envs", len(plan["seeds"]))
    if n_envs != len(plan["seeds"]):
        raise ValueError("For now n_envs must equal the number of seeds in the plan")
    attention_config = plan.get("attention", {})
    attention_enabled = attention_config.get("enabled", False)
    attention_csv = attention_config.get("output_csv")
    if attention_enabled and attention_csv is None:
        raise ValueError("attention.enabled requires attention.output_csv")

    with cfg.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for condition in plan["conditions"]:
            method = condition["method"]
            camera = condition.get("cameras", condition.get("camera", "image2"))
            camera_label = ",".join(camera) if isinstance(camera, list) else camera
            kernel_size = condition.get("kernel_size", "")
            corruption = make_corruption(method, kernel_size or 5, camera)
            condition_name = condition.get("name", corruption.name)
            print(f"\nCondition: {condition_name}")

            for task_id in plan["task_ids"]:
                env, env_preprocessor, env_postprocessor = make_task_environment(
                    plan["suite"], task_id, policy_config, n_envs=n_envs
                )
                try:
                    output_names = None
                    if cfg.save_videos:
                        output_names = [
                            f"{condition_name}_task{task_id}_seed{seed}"
                            for seed in plan["seeds"]
                        ]

                    attention_metadata = [
                        {
                            "condition": condition_name,
                            "method": method,
                            "corrupted_camera": camera_label if method != "clean" else "none",
                            "kernel_size": kernel_size,
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
                        attention=attention_enabled,
                        attention_csv=attention_csv,
                        attention_metadata=attention_metadata,
                        attention_layers=attention_config.get("layers"),
                        attention_steps=attention_config.get("steps"),
                    )
                    for result in results:
                        row = {
                            "condition": condition_name,
                            "method": method,
                            "camera": camera_label,
                            "kernel_size": kernel_size,
                            "suite": plan["suite"],
                            "task_id": task_id,
                            **result,
                        }
                        writer.writerow(row)
                        file.flush()
                        grouped_successes[(condition_name, task_id)].append(result["success"])
                        print(row)
                finally:
                    env.close()

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["condition", "task_id", "n_episodes", "successes", "success_rate"],
        )
        writer.writeheader()
        for (condition_name, task_id), successes in grouped_successes.items():
            writer.writerow(
                {
                    "condition": condition_name,
                    "task_id": task_id,
                    "n_episodes": len(successes),
                    "successes": sum(successes),
                    "success_rate": sum(successes) / len(successes),
                }
            )

    print(f"\nPer-episode results: {cfg.output_csv}")
    print(f"Success-rate summary: {summary_csv}")


if __name__ == "__main__":
    main()
