"""Measure paired SmolVLA attention on identical clean rollout observations.

For every selected clean environment state, this script evaluates three inputs:
clean, blur on the main camera, and blur on the wrist camera. The environment
advances only with the clean policy action, so all three attention measurements
share robot state, language instruction, and the uncorrupted camera.
"""

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.constants import ACTION

from attention_probe import capture_cross_attention
from corruptions import GaussianBlur, NoCorruption
from runner import _success_flags, load_policy_components, make_task_environment


SNAPSHOT_FIELDS = ["task_id", "seed", "env_step", "success", "episode_steps"]


@dataclass
class ControlledAttentionConfig:
    plan_path: Path = Path("configs/controlled_attention.json")


def load_plan(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)

    required = {
        "suite",
        "task_ids",
        "seeds",
        "max_steps",
        "observation_steps",
        "kernel_size",
        "layers",
        "attention_csv",
        "snapshots_csv",
    }
    missing = required - plan.keys()
    if missing:
        raise ValueError(f"Plan is missing keys: {', '.join(sorted(missing))}")
    if plan["kernel_size"] % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return plan


def _prepare(policy_env, raw_observation, env_preprocessor, preprocessor):
    observation = preprocess_observation(raw_observation)
    observation = add_envs_task(policy_env, observation)
    observation = env_preprocessor(observation)
    return preprocessor(observation)


def _record_three_conditions(
    *,
    policy,
    env,
    raw_observation,
    env_preprocessor,
    preprocessor,
    plan,
    task_id: int,
    seed: int,
    step: int,
):
    variants = [
        ("clean", "clean", "none", "", NoCorruption()),
        ("blur_51_0", "blur", "image", plan["kernel_size"], GaussianBlur("image", plan["kernel_size"])),
        ("blur_51_1", "blur", "image2", plan["kernel_size"], GaussianBlur("image2", plan["kernel_size"])),
    ]

    for condition, method, corrupted_camera, kernel_size, corruption in variants:
        variant = deepcopy(raw_observation)
        corruption.reset()
        variant = corruption.apply(variant, step)
        observation = _prepare(env, variant, env_preprocessor, preprocessor)
        capture_cross_attention(
            policy=policy,
            observation=observation,
            csv_path=plan["attention_csv"],
            metadata={
                "condition": condition,
                "method": method,
                "corrupted_camera": corrupted_camera,
                "kernel_size": kernel_size,
                "suite": plan["suite"],
                "task_id": task_id,
                "seed": seed,
                "env_step": step,
            },
            layers=plan["layers"],
            noise_seed=0,
        )


@parser.wrap()
def main(cfg: ControlledAttentionConfig):
    plan = load_plan(cfg.plan_path)
    attention_csv = Path(plan["attention_csv"])
    snapshots_csv = Path(plan["snapshots_csv"])
    if attention_csv.exists() or snapshots_csv.exists():
        raise FileExistsError(
            "Controlled-attention output already exists. Choose new paths or move the old run first."
        )
    attention_csv.parent.mkdir(parents=True, exist_ok=True)
    snapshots_csv.parent.mkdir(parents=True, exist_ok=True)

    policy, policy_config, preprocessor, postprocessor = load_policy_components()
    snapshot_rows = []
    selected_steps = set(plan["observation_steps"])

    for task_id in plan["task_ids"]:
        env, env_preprocessor, env_postprocessor = make_task_environment(
            plan["suite"], task_id, policy_config, n_envs=1
        )
        try:
            for seed in plan["seeds"]:
                print(f"task_id={task_id}, seed={seed}")
                policy.reset()
                raw_observation, _ = env.reset(seed=[seed])
                done = False
                success = False
                captured_steps = []

                for step in range(plan["max_steps"]):
                    if done:
                        break

                    if step in selected_steps:
                        _record_three_conditions(
                            policy=policy,
                            env=env,
                            raw_observation=raw_observation,
                            env_preprocessor=env_preprocessor,
                            preprocessor=preprocessor,
                            plan=plan,
                            task_id=task_id,
                            seed=seed,
                            step=step,
                        )
                        captured_steps.append(step)

                    clean_observation = _prepare(
                        env, raw_observation, env_preprocessor, preprocessor
                    )
                    with torch.inference_mode():
                        action = policy.select_action(clean_observation)
                    action = postprocessor(action)
                    action = env_postprocessor({ACTION: action})[ACTION]

                    raw_observation, _, terminated, truncated, info = env.step(action.cpu().numpy())
                    success |= bool(_success_flags(info, 1)[0])
                    done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])

                for captured_step in captured_steps:
                    snapshot_rows.append(
                        {
                            "task_id": task_id,
                            "seed": seed,
                            "env_step": captured_step,
                            "success": success,
                            "episode_steps": step + 1,
                        }
                    )
        finally:
            env.close()

    with snapshots_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(snapshot_rows)

    print(f"Attention: {attention_csv}")
    print(f"Snapshots: {snapshots_csv} ({len(snapshot_rows)} states)")


if __name__ == "__main__":
    main()
