import argparse
from pathlib import Path

import torch
from model.clip import CLIP
from data.data_utils import tokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, lr_scheduler
from dalle2_dataset import (
    add_dataset_arguments,
    config_from_args,
    get_test_set,
    get_train_set,
)

def train_clip(config):
    if config.using_pretrained_clip:
        raise ValueError(
            "train_clip() is only for the custom CLIP. Remove "
            "--using-pre-CLIP or start training from train_prior.py."
        )
    clip = CLIP(config).to(config.device)

    # Loading train and validation sets
    train_set, mean, std = get_train_set(config, augment_data=config.clip.augment_data)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.clip.batch_size,
        num_workers=config.clip.num_workers,
        drop_last=True,
        pin_memory=config.device.type == "cuda",
    )

    if config.clip.validate:
        val_set = get_test_set(config, mean=mean, std=std)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=config.clip.batch_size, num_workers=config.clip.num_workers)

        # Getting dataset captions to compare images to during validation
        if config.clip.get_val_accuracy:
            val_captions = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[0] for x in val_set.captions.values()]).to(config.device)
            val_masks = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[1] for x in val_set.captions.values()]).to(config.device)

    if config.clip.weight_decay == 0:
        optimizer = Adam(clip.parameters(), lr=config.clip.lr)
    else:
        optimizer = AdamW(clip.parameters(), lr=config.clip.lr, weight_decay=config.clip.weight_decay)

    if config.clip.warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.clip.warmup_epochs), end_factor=1.0, total_iters=max(1, config.clip.warmup_epochs), last_epoch=-1)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.clip.epochs - config.clip.warmup_epochs), eta_min=config.clip.lr_min)

    best_loss = float('inf')

    for epoch in range(config.clip.epochs):
        # Training
        clip.train()
        train_loss = 0.0
        for batch in train_loader:
            images, captions, masks = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
            optimizer.zero_grad()
            loss = clip(images, captions, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip.parameters(), max_norm=config.clip.grad_max_norm)
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        # Update learning rate scheduler
        if epoch < config.clip.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        # Validation
        if config.clip.validate:
            clip.eval()
            val_loss = 0.0
            correct, total = 0,0
            with torch.inference_mode():
                for batch in val_loader:
                    images, captions, masks = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
                    loss = clip(images, captions, masks)
                    val_loss += loss.item()

                    if config.clip.get_val_accuracy:
                        # Calculating the probabilities for each caption and choosing the caption with the highest probability
                        image_features = torch.nn.functional.normalize(clip.image_encoder(images), dim=-1)
                        text_features = torch.nn.functional.normalize(clip.text_encoder(val_captions, mask=val_masks), dim=-1)

                        # Calculating the probabilities for each caption and choosing the caption with the highest probability
                        similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)
                        _, indices = torch.max(similarity, 1)
                        pred_captions = val_captions[indices].to(config.device)

                        # Comparing predicted caption with actual caption
                        correct += int(sum(torch.sum((pred_captions == captions), dim=1) // len(pred_captions[0])))
                        total += len(captions)

            val_loss = val_loss / len(val_loader)

            # Saves model if it performed better than the previous best
            if val_loss <= best_loss:
                best_loss = val_loss
                Path(config.clip.model_location).parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(clip.state_dict(), config.clip.model_location)

            # Print out metrics
            if config.clip.get_val_accuracy:
                print(f"[Epoch {epoch+1}/{config.clip.epochs}] Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f} | Validation Accuracy: {100 * correct / total:.2f}")
            else:
                print(f"[Epoch {epoch+1}/{config.clip.epochs}] Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}")
        else:
            # Save model
            Path(config.clip.model_location).parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(clip.state_dict(), config.clip.model_location)

            # Print out metrics
            print(f"[Epoch {epoch+1}/{config.clip.epochs}] Training Loss: {train_loss:.3f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the DALL-E 2 CLIP stage")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
    print("Using dataset:", config.dataset, "from", config.data_location)
    train_clip(config)
