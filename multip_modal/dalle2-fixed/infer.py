"""Generate images with the trained educational DALL-E 2 pipeline."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from dalle2_dataset import add_dataset_arguments, config_from_args
from data.data_utils import tokenizer
from model.decoder import sample_image


DEFAULT_PROMPTS = {
    "fashion_mnist": "An image of a sneaker",
    "flickr8k": "a dog running through green grass",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DALL-E 2 inference")
    add_dataset_arguments(parser)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_images.png"),
    )
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-images must be positive")

    config = config_from_args(args)
    prompt_text = args.prompt or DEFAULT_PROMPTS[config.dataset]
    required_checkpoints = [
        config.prior.model_location,
        config.decoder.model_location,
    ]
    if not config.using_pretrained_clip:
        required_checkpoints.insert(0, config.clip.model_location)
    missing = [
        path for path in required_checkpoints if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing trained checkpoint(s): {missing}")

    caption, mask = tokenizer(
        prompt_text,
        text_seq_length=config.text_seq_length,
    )
    captions = caption.unsqueeze(0).repeat(args.num_images, 1).to(config.device)
    masks = mask.unsqueeze(0).repeat(args.num_images, 1).to(config.device)

    images = sample_image(config, captions, masks)
    mean = torch.tensor(config.train_mean, device=images.device)[None, :, None, None]
    std = torch.tensor(config.train_std, device=images.device)[None, :, None, None]
    images = (images * std + mean).clamp(0.0, 1.0)
    saturation = ((images <= 1e-4) | (images >= 1.0 - 1e-4)).float().mean()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.cpu(), output, nrow=min(args.num_images, 4))
    print(f"Prompt: {prompt_text}")
    print(f"Saved {args.num_images} image(s) to: {output}")
    print(f"Final pixel saturation: {100.0 * saturation.item():.2f}%")
    if saturation.item() > 0.25:
        print(
            "WARNING: more than 25% of pixels are clipped; the sampler or "
            "decoder may still be unstable."
        )


if __name__ == "__main__":
    main()

"""

  Key changes:

  - Added multip_modal/dalle2-fixed/dalle2_dataset.py:1.
      - Supports fashion_mnist and flickr8k.
      - Loads Flickr8k from local Parquet files.
      - Uses RGB 64×64 images.
      - Randomly selects one of five captions per training image.
      - Uses deterministic captions for validation.
      - Adds --dataset, --data-dir, and --device.

  - Updated all training stages:
      - multip_modal/dalle2-fixed/train_clip.py
      - multip_modal/dalle2-fixed/train_prior.py
      - multip_modal/dalle2-fixed/train_decoder.py

  - Fixed important implementation problems:
      - Proper multi-step reverse diffusion in the prior.
      - Correct posterior standard deviation during sampling.
      - Correct CUDA placement for timestep tensors.
      - Frozen CLIP/prior models remain in evaluation mode.
      - Decoder training uses the ground-truth CLIP image embedding.
      - Decoder inference samples the prior only once per trajectory.
      - Validation no longer constructs computation graphs.
      - Fixed tokenizer truncation and EOS handling.
      - Fixed mutable U-Net skip-connection state.
      - Added normalized pixel-range handling.
      - Inference now saves images instead of requiring a graphical display.

  - Updated multip_modal/dalle2-fixed/infer.py with prompt, dataset, output, device, and image-count arguments.
  - Added the theory and implementation note:
    multip_modal/dalle2-fixed/understand_dalle2.md:1

"""
