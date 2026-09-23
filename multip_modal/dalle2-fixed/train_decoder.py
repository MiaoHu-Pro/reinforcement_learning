import argparse
from pathlib import Path

import torch
import torch.nn as nn
from os.path import isfile
from train_clip import train_clip
from train_prior import train_prior
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, lr_scheduler
from dalle2_dataset import (
    add_dataset_arguments,
    config_from_args,
    get_test_set,
    get_train_set,
)
from model.decoder import Decoder, sample_plot_image
from data.data_utils import get_schedule_values, forward_diffusion, tokenizer

def train_decoder(config):
    train_set, mean, std = get_train_set(config, augment_data=config.decoder.augment_data)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.decoder.batch_size,
        num_workers=config.decoder.num_workers,
        pin_memory=config.device.type == "cuda",
        persistent_workers=config.decoder.num_workers > 0,
    )

    if config.decoder.validate:
        val_set = get_test_set(config, mean=mean, std=std)
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            batch_size=config.decoder.batch_size,
            num_workers=config.decoder.num_workers,
            pin_memory=config.device.type == "cuda",
            persistent_workers=config.decoder.num_workers > 0,
        )

    schedule_values = get_schedule_values(schedule=config.decoder.schedule, max_time=config.decoder.max_time, device=config.device)

    decoder = Decoder(config).to(config.device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in decoder.parameters()
        if parameter.requires_grad
    )
    use_bf16 = (
        config.device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    print(
        "Decoder architecture:",
        "large U-Net" if config.large_unet else "default U-Net",
    )
    print(
        "Decoder channels:",
        [
            config.decoder.model_channels * ratio
            for ratio in config.decoder.channel_ratios
        ],
    )
    print(f"Trainable decoder parameters: {trainable_parameters:,}")
    print(f"Decoder batch size: {config.decoder.batch_size}")
    print(f"BF16 autocast: {use_bf16}")

    if config.decoder.weight_decay == 0:
        optimizer = Adam(decoder.parameters(), lr=config.decoder.lr)
    else:
        optimizer = AdamW(decoder.parameters(), lr=config.decoder.lr, weight_decay=config.decoder.weight_decay)

    if config.decoder.warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.decoder.warmup_epochs), end_factor=1.0, total_iters=max(1, config.decoder.warmup_epochs), last_epoch=-1)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.decoder.epochs - config.decoder.warmup_epochs), eta_min=config.decoder.lr_min)

    if config.decoder.sample_after_epoch:
        sample_texts = train_set.sample_texts()
        sample_captions = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[0] for x in sample_texts]).to(config.device)
        sample_masks = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[1] for x in sample_texts]).to(config.device)

    best_loss = float('inf')
    for epoch in range(config.decoder.epochs):
        # Training
        decoder.train()
        training_loss = 0.0
        for batch in train_loader:
            image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
            optimizer.zero_grad(set_to_none=True)

            # Calculating Loss
            # 采样时间步t
            timesteps = torch.randint(0, config.decoder.max_time, (image.shape[0],), device=config.device, dtype=torch.long)
            # 前向扩散，(x_0, t) ---> (x_t, 噪声)
            noisy_image, noise = forward_diffusion(image, schedule_values, timesteps)
            with torch.autocast(
                device_type=config.device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                # During decoder training DALL-E 2 conditions on the ground-truth
                # image's frozen CLIP embedding, not a newly sampled prior output.
                with torch.no_grad():
                    image_embedding = decoder.prior.clip.image_encoder(image)
                pred_noise = decoder(
                    noisy_image,
                    timesteps,
                    caption,
                    mask,
                    image_embedding=image_embedding,
                )
                loss = nn.functional.mse_loss(
                    pred_noise.float(), noise.float()
                )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=config.decoder.grad_max_norm)
            optimizer.step()
            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)

        if epoch < config.decoder.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        # Validation
        if config.decoder.validate:
            decoder.eval()
            validation_loss = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)

                    timesteps = torch.randint(0, config.decoder.max_time, (image.shape[0],), device=config.device, dtype=torch.long)
                    noisy_image, noise = forward_diffusion(image, schedule_values, timesteps)
                    with torch.autocast(
                        device_type=config.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        image_embedding = decoder.prior.clip.image_encoder(image)
                        pred_noise = decoder(
                            noisy_image,
                            timesteps,
                            caption,
                            mask,
                            image_embedding=image_embedding,
                        )
                        loss = nn.functional.mse_loss(
                            pred_noise.float(), noise.float()
                        )
                    validation_loss += loss.item()

            validation_loss = validation_loss / len(val_loader)

            if validation_loss <= best_loss:
                best_loss = validation_loss
                Path(config.decoder.model_location).parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(decoder.state_dict(), config.decoder.model_location)

            print(f"[Epoch {epoch + 1}/{config.decoder.epochs}] Training Loss: {training_loss:.5f} | Validation Loss: {validation_loss:.5f}")
        else:
            Path(config.decoder.model_location).parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(decoder.state_dict(), config.decoder.model_location)
            print(f"[Epoch {epoch + 1}/{config.decoder.epochs}] Training Loss: {training_loss:.5f}")

        if config.decoder.sample_after_epoch:
            caption = sample_captions[None, (epoch % len(sample_captions))]
            mask = sample_masks[None, (epoch % len(sample_masks))]
            sample_plot_image(config, caption, mask, schedule_values=schedule_values, decoder=decoder)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the DALL-E 2 decoder stage")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    if not config.using_pretrained_clip and not isfile(config.clip.model_location):
        print("CLIP model has not been trained. Training CLIP...")
        print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
        train_clip(config)

    if not isfile(config.prior.model_location):
        print("Prior model has not been trained. Training Prior...")
        print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
        train_prior(config)

    print("Training Decoder...")
    if config.using_pretrained_clip:
        print("Using frozen pretrained CLIP:", config.pretrained_clip_path)
    print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
    print("Using dataset:", config.dataset, "from", config.data_location)
    train_decoder(config)
