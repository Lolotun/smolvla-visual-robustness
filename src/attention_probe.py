"""Record SmolVLA expert-to-camera cross-attention in a CSV file.

The probe patches one *loaded policy instance* temporarily. It does not modify
LeRobot files in ``.venv`` and restores the original model methods afterwards.
"""

from __future__ import annotations

import csv
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


CSV_FIELDS = [
    "condition",
    "method",
    "corrupted_camera",
    "kernel_size",
    "suite",
    "task_id",
    "seed",
    "env_step",
    "batch_index",
    "layer",
    "denoise_step",
    "head",
    "camera",
    "attention_mass",
]


def _camera_token_ranges(policy, observation: dict[str, torch.Tensor]) -> list[tuple[str, int, int]]:
    """Return [camera name, first token, one-past-last token] in the prefix."""
    with torch.inference_mode():
        images, _ = policy.prepare_images(observation)
        image_keys = [key for key in policy.config.image_features if key in observation]
        token_counts = [
            policy.model.vlm_with_expert.embed_image(image).shape[1] for image in images
        ]

    cursor = 0
    ranges = []
    for image_key, token_count in zip(image_keys, token_counts, strict=False):
        # SmolVLA optionally places two tokens before and one after every image.
        if policy.config.add_image_special_tokens:
            cursor += 2
        start = cursor
        cursor += token_count
        camera = image_key.rsplit(".", maxsplit=1)[-1]
        ranges.append((camera, start, cursor))
        if policy.config.add_image_special_tokens:
            cursor += 1
    return ranges


class _CrossAttentionRecorder(AbstractContextManager):
    """Temporarily record expert cross-attention mass for selected layers."""

    def __init__(self, policy, camera_ranges, layers: Iterable[int] | None):
        self.policy = policy
        self.camera_ranges = camera_ranges
        self.layers = None if layers is None else set(layers)
        self.rows: list[dict[str, Any]] = []
        self.current_layer: int | None = None
        self.current_denoise_step: int | None = None
        self._denoise_calls = 0

    def __enter__(self):
        model = self.policy.model
        vlm = model.vlm_with_expert
        self._original_cross = vlm.forward_cross_attn_layer
        self._original_attention = vlm.eager_attention_forward
        self._original_denoise = model.denoise_step

        def cross_wrapper(*args, **kwargs):
            layer = kwargs.get("layer_idx", args[2])
            previous_layer = self.current_layer
            self.current_layer = layer
            try:
                return self._original_cross(*args, **kwargs)
            finally:
                self.current_layer = previous_layer

        def attention_wrapper(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            # Expert cross-attention has action tokens as Q and a longer visual
            # prefix as K. Prefix self-attention and expert self-attention are
            # intentionally excluded.
            is_expert_cross_attention = (
                self.current_layer is not None
                and query_states.shape[1] == self.policy.config.chunk_size
                and key_states.shape[1] > query_states.shape[1]
                and (self.layers is None or self.current_layer in self.layers)
            )
            if is_expert_cross_attention:
                self._record_probs(attention_mask, batch_size, head_dim, query_states, key_states)
            return self._original_attention(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )

        def denoise_wrapper(*args, **kwargs):
            self.current_denoise_step = self._denoise_calls
            self._denoise_calls += 1
            return self._original_denoise(*args, **kwargs)

        # Instance attributes are intentionally used: these wrappers affect only
        # this loaded policy, and __exit__ restores its original bound methods.
        vlm.forward_cross_attn_layer = cross_wrapper
        vlm.eager_attention_forward = attention_wrapper
        model.denoise_step = denoise_wrapper
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        vlm = self.policy.model.vlm_with_expert
        vlm.forward_cross_attn_layer = self._original_cross
        vlm.eager_attention_forward = self._original_attention
        self.policy.model.denoise_step = self._original_denoise
        return False

    def _record_probs(self, attention_mask, batch_size, head_dim, query_states, key_states) -> None:
        """Recompute probs exactly as eager_attention_forward, then aggregate."""
        vlm = self.policy.model.vlm_with_expert
        num_heads = vlm.num_attention_heads
        num_kv_heads = vlm.num_key_value_heads
        groups = num_heads // num_kv_heads
        key_length = key_states.shape[1]

        expanded_keys = key_states[:, :, :, None, :].expand(
            batch_size, key_length, num_kv_heads, groups, head_dim
        )
        expanded_keys = expanded_keys.reshape(batch_size, key_length, num_heads, head_dim)

        queries = query_states.to(dtype=torch.float32).transpose(1, 2)
        keys = expanded_keys.to(dtype=torch.float32).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(2, 3)) * (head_dim**-0.5)
        masked_scores = torch.where(
            attention_mask[:, None, :, :], scores, torch.finfo(scores.dtype).min
        )
        probs = torch.softmax(masked_scores, dim=-1)

        for camera, start, end in self.camera_ranges:
            # Sum camera tokens first: it is attention mass in [0, 1] for every
            # batch item, head, and action query. Then average action queries.
            mass_by_head = probs[..., start:end].sum(dim=-1).mean(dim=-1).detach().cpu()
            for batch_index in range(batch_size):
                for head, mass in enumerate(mass_by_head[batch_index].tolist()):
                    self.rows.append(
                        {
                            "batch_index": batch_index,
                            "layer": self.current_layer,
                            "denoise_step": self.current_denoise_step,
                            "head": head,
                            "camera": camera,
                            "attention_mass": mass,
                        }
                    )


def _append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def capture_cross_attention(
    policy,
    observation: dict[str, torch.Tensor],
    csv_path: str | Path,
    metadata: Mapping[str, Any] | list[Mapping[str, Any]],
    layers: Iterable[int] | None = None,
    noise_seed: int = 0,
) -> torch.Tensor:
    """Run one diagnostic action-chunk prediction and append attention rows.

    ``observation`` must already have passed the normal LeRobot preprocessor.
    For a clean-vs-blur comparison, call this on the same environment state and
    retain the default ``noise_seed`` for both conditions. ``layers=[15]`` is a
    compact first probe; ``layers=None`` records every expert cross-attention
    layer.
    """
    camera_ranges = _camera_token_ranges(policy, observation)
    if not camera_ranges:
        raise ValueError("No policy image features were found in observation")

    batch_size = observation[next(iter(policy.config.image_features))].shape[0]
    if isinstance(metadata, Mapping):
        metadata_per_item = [metadata] * batch_size
    else:
        metadata_per_item = list(metadata)
    if len(metadata_per_item) != batch_size:
        raise ValueError("metadata must provide one mapping per batch item")

    for item in metadata_per_item:
        missing = set(CSV_FIELDS[:8]) - item.keys()
        if missing:
            raise ValueError(f"metadata is missing: {', '.join(sorted(missing))}")

    first_image = observation[next(iter(policy.config.image_features))]
    generator = torch.Generator(device=first_image.device).manual_seed(noise_seed)
    noise = torch.randn(
        (batch_size, policy.config.chunk_size, policy.config.max_action_dim),
        device=first_image.device,
        dtype=first_image.dtype,
        generator=generator,
    )

    with _CrossAttentionRecorder(policy, camera_ranges, layers) as recorder:
        with torch.inference_mode():
            action_chunk = policy.predict_action_chunk(observation, noise=noise)

    rows = []
    for record in recorder.rows:
        row = {**metadata_per_item[record["batch_index"]], **record}
        rows.append({field: row[field] for field in CSV_FIELDS})
    _append_rows(Path(csv_path), rows)
    return action_chunk
