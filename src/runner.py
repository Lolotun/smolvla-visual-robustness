"""One LIBERO rollout, with reusable policy and environment builders."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.io_utils import write_video

from attention_probe import capture_cross_attention
from corruptions import NoCorruption, VisualCorruption, make_corruption


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
DEVICE = "cuda"
VIDEO_FPS = 20


def load_policy_components():
    """Load SmolVLA and its processors once; reuse them across all episodes."""
    reference_env_config = LiberoEnv(task="libero_spatial", task_ids=[0])

    policy_config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    policy_config.pretrained_path = Path(CHECKPOINT)
    policy_config.device = DEVICE

    policy = make_policy(cfg=policy_config, env_cfg=reference_env_config)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )
    return policy, policy_config, preprocessor, postprocessor


def make_task_environment(task: str, task_id: int, policy_config, n_envs: int = 1):
    """Create a vectorized LIBERO task environment and its processors."""
    env_config = LiberoEnv(task=task, task_ids=[task_id])
    environments = make_env(env_config, n_envs=n_envs)
    env = environments[task][task_id]

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config,
        policy_cfg=policy_config,
    )
    return env, env_preprocessor, env_postprocessor


def _success_flags(info: dict, n_envs: int) -> np.ndarray:
    """Extract per-environment success flags from Gymnasium vector ``info``."""
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return np.asarray(final_info["is_success"], dtype=bool)

    # Fallback for vector environments that expose one dict per completed env.
    if isinstance(final_info, np.ndarray):
        return np.asarray(
            [bool(item.get("is_success", False)) if item is not None else False for item in final_info],
            dtype=bool,
        )
    return np.zeros(n_envs, dtype=bool)


def run_batch(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seeds: list[int],
    max_steps: int | None = None,
    corruption: VisualCorruption | None = None,
    output_names: list[str] | None = None,
    videos_dir: str | Path | None = None,
    attention: bool = False,
    attention_csv: str | Path | None = None,
    attention_metadata: list[dict] | None = None,
    attention_layers: list[int] | None = None,
    attention_steps: list[int] | None = None,
):
    """Run one rollout per seed simultaneously and return one result per seed.

    ``env.num_envs`` must equal ``len(seeds)``. The policy receives one batch
    of observations per control step, while each vector slot keeps its own
    reward, completion flag, and video.
    """
    if env.num_envs != len(seeds):
        raise ValueError(f"env has {env.num_envs} slots but received {len(seeds)} seeds")
    if output_names is not None and len(output_names) != env.num_envs:
        raise ValueError("output_names must contain one name per vector-environment slot")
    if attention and (attention_csv is None or attention_metadata is None):
        raise ValueError("attention requires attention_csv and attention_metadata")
    if attention_metadata is not None and len(attention_metadata) != env.num_envs:
        raise ValueError("attention_metadata must contain one mapping per vector-environment slot")

    policy.reset()
    corruption = corruption or NoCorruption()
    corruption.reset()
    raw_observation, _ = env.reset(seed=seeds)

    n_envs = env.num_envs
    done = np.zeros(n_envs, dtype=bool)
    successes = np.zeros(n_envs, dtype=bool)
    total_rewards = np.zeros(n_envs, dtype=float)
    max_rewards = np.full(n_envs, -np.inf, dtype=float)
    episode_steps = np.zeros(n_envs, dtype=int)
    main_frames = [[] for _ in range(n_envs)]
    wrist_frames = [[] for _ in range(n_envs)]

    env_max_steps = env.call("_max_episode_steps")[0]
    max_steps = min(max_steps, env_max_steps) if max_steps is not None else env_max_steps
    progress = trange(max_steps, desc="Running batched rollout", leave=False)
    attention_steps = {0} if attention_steps is None else set(attention_steps)

    for step in progress:
        if done.all():
            break

        active = ~done
        raw_observation = corruption.apply(raw_observation, step)

        if output_names is not None:
            for index in np.flatnonzero(active):
                # Same orientation as the images seen by SmolVLA.
                main_frames[index].append(
                    np.flip(raw_observation["pixels"]["image"][index], axis=(0, 1)).copy()
                )
                wrist_frames[index].append(
                    np.flip(raw_observation["pixels"]["image2"][index], axis=(0, 1)).copy()
                )

        observation = preprocess_observation(raw_observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        if attention and step in attention_steps:
            capture_cross_attention(
                policy=policy,
                observation=observation,
                csv_path=attention_csv,
                metadata=[{**item, "env_step": step} for item in attention_metadata],
                layers=attention_layers,
            )

        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]

        raw_observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        reward = np.asarray(reward, dtype=float)
        total_rewards[active] += reward[active]
        max_rewards[active] = np.maximum(max_rewards[active], reward[active])
        episode_steps[active] += 1

        successes |= _success_flags(info, n_envs)
        done |= np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        progress.set_postfix({"success": f"{successes.sum()}/{n_envs}"})

    if output_names is not None:
        videos_dir = Path(videos_dir) if videos_dir is not None else Path("results/videos") / corruption.name
        videos_dir.mkdir(parents=True, exist_ok=True)
        for index, output_name in enumerate(output_names):
            if main_frames[index]:
                write_video(
                    str(videos_dir / f"{output_name}_main.mp4"),
                    np.stack(main_frames[index]),
                    fps=VIDEO_FPS,
                )
                write_video(
                    str(videos_dir / f"{output_name}_wrist.mp4"),
                    np.stack(wrist_frames[index]),
                    fps=VIDEO_FPS,
                )

    return [
        {
            "seed": seed,
            "success": bool(successes[index]),
            "steps": int(episode_steps[index]),
            "total_reward": float(total_rewards[index]),
            "max_reward": float(max_rewards[index]),
        }
        for index, seed in enumerate(seeds)
    ]


def run_episode(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    max_steps: int | None = None,
    corruption: VisualCorruption | None = None,
    output_name: str | None = None,
    videos_dir: str | Path | None = None,
    return_observations: bool = False,
    attention: bool = False,
    attention_csv: str | Path | None = None,
    attention_metadata: dict | None = None,
    attention_layers: list[int] | None = None,
    attention_steps: list[int] | None = None,
):
    """Run one episode and return the metrics needed for the experiment table.

    Set ``output_name`` to save ``<name>_main.mp4`` and ``<name>_wrist.mp4``.
    """
    policy.reset()
    corruption = corruption or NoCorruption()
    if attention and (attention_csv is None or attention_metadata is None):
        raise ValueError("attention requires attention_csv and attention_metadata")
    corruption.reset()
    raw_observation, info = env.reset(seed=[seed])

    all_observations = []
    main_frames = []
    wrist_frames = []
    total_reward = 0.0
    max_reward = float("-inf")
    success = False

    done = np.array([False] * env.num_envs)
    env_max_steps = env.call("_max_episode_steps")[0]
    max_steps = min(max_steps, env_max_steps) if max_steps is not None else env_max_steps
    progress = trange(max_steps, desc="Running rollout", leave=False)
    attention_steps = {0} if attention_steps is None else set(attention_steps)

    for step in progress:
        if np.all(done):
            break

        # All corruptions enter here while observations are still NumPy images.
        raw_observation = corruption.apply(raw_observation, step)

        if output_name is not None:
            # LIBERO returns raw cameras upside down. LiberoProcessorStep flips
            # both axes before SmolVLA sees them, so do the same for videos.
            main_frames.append(np.flip(raw_observation["pixels"]["image"][0], axis=(0, 1)).copy())
            wrist_frames.append(np.flip(raw_observation["pixels"]["image2"][0], axis=(0, 1)).copy())

        observation = preprocess_observation(raw_observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        if attention and step in attention_steps:
            capture_cross_attention(
                policy=policy,
                observation=observation,
                csv_path=attention_csv,
                metadata={**attention_metadata, "env_step": step},
                layers=attention_layers,
            )

        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]

        action_numpy = action.cpu().numpy()
        raw_observation, reward, terminated, truncated, info = env.step(action_numpy)

        total_reward += float(reward[0])
        max_reward = max(max_reward, float(reward[0]))
        done = terminated | truncated | done

        if "final_info" in info:
            success = bool(info["final_info"]["is_success"][0])

        progress.set_postfix({"success": success})

    if return_observations:
        all_observations.append(deepcopy(preprocess_observation(raw_observation)))

    if output_name is not None:
        videos_dir = Path(videos_dir) if videos_dir is not None else Path("results/videos") / corruption.name
        videos_dir.mkdir(parents=True, exist_ok=True)
        write_video(str(videos_dir / f"{output_name}_main.mp4"), np.stack(main_frames), fps=VIDEO_FPS)
        write_video(str(videos_dir / f"{output_name}_wrist.mp4"), np.stack(wrist_frames), fps=VIDEO_FPS)

    result = {
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "total_reward": total_reward,
        "max_reward": max_reward,
    }
    if return_observations:
        result[OBS_STR] = all_observations
    return result


@dataclass
class RunConfig:
    task: str = "libero_spatial"
    task_id: int = 0
    seed: int = 42
    corruption: str = "clean"
    camera: str = "image2"
    kernel_size: int = 5
    freeze_period: int = 5
    output_name: str = "episode"


@parser.wrap()
def main(cfg: RunConfig):
    policy, policy_config, preprocessor, postprocessor = load_policy_components()
    corruption = make_corruption(
        cfg.corruption,
        kernel_size=cfg.kernel_size,
        camera=cfg.camera,
        period=cfg.freeze_period,
    )
    env, env_preprocessor, env_postprocessor = make_task_environment(
        cfg.task, cfg.task_id, policy_config
    )
    try:
        result = run_episode(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seed=cfg.seed,
            corruption=corruption,
            output_name=cfg.output_name,
        )
        print(result)
    finally:
        env.close()


if __name__ == "__main__":
    main()
