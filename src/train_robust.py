#!/usr/bin/env python
"""Small, editable SmolVLA fine-tuning entry point.

This keeps the useful parts of ``lerobot-train`` (dataset loading, pretrained
processors, native PEFT/LoRA, optimisation and checkpoints) while exposing one
local hook: ``corrupt_training_batch``.  The hook is intentionally an identity
transform for now, so a first LoRA smoke run is equivalent to normal LeRobot
fine-tuning.  Add visual corruptions there only after the LoRA baseline works.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


@dataclass
class RobustAugmentationConfig:
    """Per-sample camera corruption distribution used only during training."""

    enabled: bool = True
    blur_kernel_size: int = 31
    clean_probability: float = 0.50
    blur_main_probability: float = 0.15
    blur_wrist_probability: float = 0.15
    blackout_main_probability: float = 0.10
    blackout_wrist_probability: float = 0.10

    def validate(self) -> None:
        probabilities = [
            self.clean_probability,
            self.blur_main_probability,
            self.blur_wrist_probability,
            self.blackout_main_probability,
            self.blackout_wrist_probability,
        ]
        if any(p < 0 for p in probabilities) or not abs(sum(probabilities) - 1.0) < 1e-6:
            raise ValueError("Robust augmentation probabilities must be non-negative and sum to 1.")
        if self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be an odd integer >= 3.")


@dataclass
class RobustTrainConfig(TrainPipelineConfig):
    robust_augmentation: RobustAugmentationConfig = field(default_factory=RobustAugmentationConfig)


def gaussian_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Blur a B×C×H×W tensor with a Gaussian kernel directly on its device."""

    if images.ndim != 4:
        raise ValueError(f"Expected batched images with shape (B, C, H, W), got {tuple(images.shape)}")

    # Same conventional sigma heuristic used when Gaussian blur receives only a kernel size.
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    coordinates = torch.arange(kernel_size, device=images.device, dtype=images.dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(images.shape[1], 1, kernel_size, kernel_size)

    padded = F.pad(images, (kernel_size // 2,) * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=images.shape[1])


def corrupt_training_batch(batch: dict, cfg: RobustAugmentationConfig) -> dict:
    """Apply training-only image corruption immediately before ``policy.forward``.

    The input has already passed LeRobot's preprocessor and lives on the model
    device.  Its image keys are ``observation.images.image`` and
    ``observation.images.image2``.  Keep actions untouched.

    One condition is sampled independently for every item in the batch:
    clean (50%), blur main (15%), blur wrist (15%), blackout main (10%), or
    blackout wrist (10%).  The action supervision always stays clean.
    """

    if not cfg.enabled:
        return batch

    cfg.validate()
    main_key = "observation.images.image"
    wrist_key = "observation.images.image2"
    if main_key not in batch or wrist_key not in batch:
        raise KeyError(f"Expected both {main_key!r} and {wrist_key!r} in the training batch.")

    main = batch[main_key]
    wrist = batch[wrist_key]
    if main.ndim != 4 or wrist.ndim != 4:
        raise ValueError("SmolVLA training augmentation expects image tensors shaped (B, C, H, W).")
    if main.shape[0] != wrist.shape[0]:
        raise ValueError("Main and wrist camera tensors must have the same batch size.")

    # Copy only the two tensors that may be modified. All labels remain untouched.
    batch = dict(batch)
    main = main.clone()
    wrist = wrist.clone()
    draw = torch.rand(main.shape[0], device=main.device)

    blur_main_end = cfg.clean_probability + cfg.blur_main_probability
    blur_wrist_end = blur_main_end + cfg.blur_wrist_probability
    blackout_main_end = blur_wrist_end + cfg.blackout_main_probability

    blur_main_mask = (draw >= cfg.clean_probability) & (draw < blur_main_end)
    blur_wrist_mask = (draw >= blur_main_end) & (draw < blur_wrist_end)
    blackout_main_mask = (draw >= blur_wrist_end) & (draw < blackout_main_end)
    blackout_wrist_mask = draw >= blackout_main_end

    if blur_main_mask.any():
        main[blur_main_mask] = gaussian_blur(main[blur_main_mask], cfg.blur_kernel_size)
    if blur_wrist_mask.any():
        wrist[blur_wrist_mask] = gaussian_blur(wrist[blur_wrist_mask], cfg.blur_kernel_size)
    main[blackout_main_mask] = 0
    wrist[blackout_wrist_mask] = 0

    batch[main_key] = main
    batch[wrist_key] = wrist
    return batch


def build_processors(cfg: TrainPipelineConfig, policy, dataset, device):
    """Use the same processor construction as the official training script."""

    processor_kwargs = {}
    postprocessor_kwargs = {}

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }
    else:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )


@parser.wrap()
def train(cfg: RobustTrainConfig):
    """Fine-tune one policy on an offline dataset and save loadable checkpoints."""

    if cfg.resume:
        raise NotImplementedError("Resume is omitted from this small local training script.")
    if cfg.env is not None or cfg.eval_freq > 0:
        raise ValueError("Evaluate checkpoints separately with src/experiment.py.")

    cfg.validate()
    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    init_logging(accelerator=accelerator)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    # Native LeRobot PEFT.  SmolVLA supplies its own default target modules
    # when --peft is present and --peft.target_modules is omitted.
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    preprocessor, postprocessor = build_processors(cfg, policy, dataset, accelerator.device)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming,
        num_workers=cfg.num_workers,
        pin_memory=accelerator.device.type == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    batches = cycle(dataloader)

    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        metrics,
        accelerator=accelerator,
    )

    if accelerator.is_main_process:
        learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logging.info("Dataset: %s episodes, %s frames", dataset.num_episodes, dataset.num_frames)
        logging.info("Trainable parameters: %s", format_big_number(learnable))
        logging.info("Output: %s", cfg.output_dir)

    progress = tqdm(total=cfg.steps, desc="Training", unit="step", disable=not accelerator.is_main_process)
    for step in range(1, cfg.steps + 1):
        load_start = time.perf_counter()
        batch = preprocessor(next(batches))
        batch = corrupt_training_batch(batch, cfg.robust_augmentation)
        tracker.dataloading_s = time.perf_counter() - load_start

        tracker, _ = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        tracker.step()
        progress.update(1)

        if accelerator.is_main_process and cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info(tracker)
            tracker.reset_averages()

        should_save = step == cfg.steps or (cfg.save_freq > 0 and step % cfg.save_freq == 0)
        if cfg.save_checkpoint and should_save and accelerator.is_main_process:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            logging.info("Saved checkpoint: %s", checkpoint_dir)

        accelerator.wait_for_everyone()

    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    train()
