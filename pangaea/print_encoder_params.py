import logging

import hydra
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pangaea.encoders.base import Encoder


def count_parameters(module: nn.Module) -> tuple[int, int, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def format_count(count: int) -> str:
    return f"{count:,} ({count / 1_000_000:.3f}M)"


def print_child_module_counts(encoder: Encoder, total_params: int) -> None:
    print("\nImmediate child modules:")
    for name, child in encoder.named_children():
        child_total, child_trainable, child_frozen = count_parameters(child)
        if child_total == 0:
            continue
        pct = 100 * child_total / total_params if total_params else 0
        print(
            f"  {name}: total={format_count(child_total)}, "
            f"trainable={format_count(child_trainable)}, "
            f"frozen={format_count(child_frozen)}, pct={pct:.2f}%"
        )


@hydra.main(version_base=None, config_path="../configs", config_name="encoder_params")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("encoder_params")

    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    encoder_choice = choices.get("encoder", "unknown")
    dataset_choice = choices.get("dataset", "unknown")

    encoder: Encoder = instantiate(cfg.encoder)
    if cfg.load_weights:
        encoder.load_encoder_weights(logger)

    total_params, trainable_params, frozen_params = count_parameters(encoder)

    print("=" * 80)
    print(f"Encoder config: {encoder_choice}")
    print(f"Dataset config: {dataset_choice}")
    print(f"Model name: {encoder.model_name}")
    print(f"Class: {type(encoder).__module__}.{type(encoder).__name__}")
    print(f"Input size: {encoder.input_size}")
    print(f"Input bands: {OmegaConf.to_yaml(cfg.encoder.input_bands).strip()}")
    print(f"Output layers: {encoder.output_layers}")
    print(f"Output dim: {encoder.output_dim}")
    print("-" * 80)
    print(f"Total parameters: {format_count(total_params)}")
    print(f"Trainable parameters: {format_count(trainable_params)}")
    print(f"Frozen parameters: {format_count(frozen_params)}")
    print("=" * 80)

    if cfg.print_by_module:
        print_child_module_counts(encoder, total_params)


if __name__ == "__main__":
    main()
