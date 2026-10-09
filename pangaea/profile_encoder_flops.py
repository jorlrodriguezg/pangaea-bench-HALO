import logging
from functools import partial

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


def format_count(value: int, unit: str) -> str:
    return f"{value:,} ({value / 1_000_000_000:.3f} G{unit})"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def construct_input(
    _input_res: tuple[int, ...],
    *,
    encoder: Encoder,
    temporal_size: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    height = width = encoder.input_size
    image = {
        modality: torch.zeros(
            1,
            len(bands),
            temporal_size,
            height,
            width,
            device=device,
        )
        for modality, bands in encoder.input_bands.items()
    }
    return {"img": image}


@hydra.main(version_base=None, config_path="../configs", config_name="encoder_flops")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("encoder_flops")
    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    encoder_choice = choices.get("encoder", "unknown")
    dataset_choice = choices.get("dataset", "unknown")
    decoder_choice = choices.get("decoder", "unknown")
    device = resolve_device(cfg.device)

    dataset_modalities = set(cfg.dataset.bands)
    encoder_modalities = set(cfg.encoder.input_bands)
    if not dataset_modalities & encoder_modalities:
        raise ValueError(
            f"Dataset modalities {sorted(dataset_modalities)} do not overlap with "
            f"encoder modalities {sorted(encoder_modalities)}."
        )

    encoder: Encoder = instantiate(cfg.encoder)
    if cfg.load_weights:
        encoder.load_encoder_weights(logger)

    model: Decoder = instantiate(cfg.decoder, encoder=encoder)
    model.eval().to(device)

    temporal_size = max(1, int(cfg.dataset.multi_temporal))
    input_constructor = partial(
        construct_input,
        encoder=encoder,
        temporal_size=temporal_size,
        device=device,
    )

    macs, _ = get_model_complexity_info(
        model,
        (1,),
        as_strings=False,
        input_constructor=input_constructor,
        print_per_layer_stat=cfg.print_per_layer,
        verbose=cfg.verbose,
        backend=cfg.backend,
    )
    if macs is None:
        raise RuntimeError("ptflops could not complete the forward pass.")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    shapes = {
        modality: (
            1,
            len(bands),
            temporal_size,
            encoder.input_size,
            encoder.input_size,
        )
        for modality, bands in encoder.input_bands.items()
    }
    padded_modalities = sorted(encoder_modalities - dataset_modalities)
    flops = 2 * macs

    print("=" * 80)
    print(f"Encoder config: {encoder_choice}")
    print(f"Decoder config: {decoder_choice}")
    print(f"Dataset config: {dataset_choice}")
    print(f"Device: {device}")
    print(f"Backend: {cfg.backend}")
    print(f"Input shapes (B, C, T, H, W): {shapes}")
    if padded_modalities:
        print(f"Zero-padded modalities absent from dataset: {padded_modalities}")
    print("-" * 80)
    print(f"Total parameters (encoder + classifier): {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MACs per sample: {format_count(macs, 'MAC')}")
    print(f"Estimated FLOPs per sample (2 x MACs): {format_count(flops, 'FLOP')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
