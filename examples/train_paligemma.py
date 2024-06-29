from jora import (
    ParaPaliGemmaConfig,
    generate_alpaca_dataset_gemma,
    train_lora_paligemma,
)
from pathlib import Path


def main():
    config = ParaPaliGemmaConfig(
        MODEL_PATH=str(
            Path(__file__).parent / "checkpoints" / "pt_224_128.params.f16.npz"
        ),
    )  # type: ignore

    dataset_path = Path(__file__).parent.parent / "number-ops-1" / "dataset"

    alpaca_dataset = generate_alpaca_dataset_gemma(
        str(dataset_path), "train", config, alpaca_mix=0.0
    )
    train_lora_paligemma(config, alpaca_dataset, "checkpoints")


if __name__ == "__main__":
    main()
