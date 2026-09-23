import argparse
from os.path import isfile
from pathlib import Path
from time import perf_counter

import torch
from train_clip import train_clip
from model.prior import DiffusionPrior
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, lr_scheduler
from dalle2_dataset import (
    add_dataset_arguments,
    config_from_args,
    describe_dataset,
    get_test_set,
    get_train_set,
)

def train_prior(config):
    train_set, mean, std = get_train_set(config, augment_data=config.prior.augment_data)
    describe_dataset(train_set, "prior train")
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.prior.batch_size, num_workers=config.prior.num_workers)

    if config.prior.validate:
        val_set = get_test_set(config, mean=mean, std=std)
        describe_dataset(val_set, "prior validation")
        val_loader = DataLoader(val_set, shuffle=False, batch_size=config.prior.batch_size, num_workers=config.prior.num_workers)

    prior = DiffusionPrior(config).to(config.device)
    trainable_parameters = sum(
        parameter.numel() for parameter in prior.parameters()
        if parameter.requires_grad
    )
    print("=" * 72, flush=True)
    print("Stage: diffusion-prior training", flush=True)
    print(f"Trainable parameters: {trainable_parameters:,}", flush=True)
    print(f"Epochs: {config.prior.epochs}", flush=True)
    print(f"Batch size: {config.prior.batch_size}", flush=True)
    print(f"Train batches/epoch: {len(train_loader):,}", flush=True)
    if config.prior.validate:
        print(f"Validation batches/epoch: {len(val_loader):,}", flush=True)
    print(f"Initial learning rate: {config.prior.lr:.3e}", flush=True)
    print(f"Checkpoint: {config.prior.model_location}", flush=True)
    print("=" * 72, flush=True)

    if config.prior.weight_decay == 0:
        optimizer = Adam(prior.parameters(), lr=config.prior.lr)
    else:
        optimizer = AdamW(prior.parameters(), lr=config.prior.lr, weight_decay=config.prior.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.prior.epochs - config.prior.warmup_epochs), eta_min=config.prior.lr_min)

    if config.prior.warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.prior.warmup_epochs), end_factor=1.0, total_iters=max(1, config.prior.warmup_epochs), last_epoch=-1)

    best_loss = float('inf')
    progress_every = max(1, len(train_loader) // 10)
    for epoch in range(config.prior.epochs):
        epoch_started = perf_counter()
        # Training
        prior.train()
        training_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
            optimizer.zero_grad()
            loss = prior(image, caption, masks=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=config.prior.grad_max_norm)
            optimizer.step()
            training_loss += loss.item()
            if batch_index % progress_every == 0 or batch_index == len(train_loader):
                gpu_log = ""
                if config.device.type == "cuda":
                    gpu_log = (
                        f" | GPU allocated: "
                        f"{torch.cuda.memory_allocated(config.device) / 1024**3:.1f} GiB"
                        f" | reserved: "
                        f"{torch.cuda.memory_reserved(config.device) / 1024**3:.1f} GiB"
                    )
                print(
                    f"[Prior progress] Epoch {epoch + 1}/{config.prior.epochs} "
                    f"| Batch {batch_index:,}/{len(train_loader):,} "
                    f"| Mean loss: {training_loss / batch_index:.5f} "
                    f"| Elapsed: {perf_counter() - epoch_started:.1f}s"
                    f"{gpu_log}",
                    flush=True,
                )

        training_loss = training_loss / len(train_loader)

        if epoch < config.prior.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        # Validation
        if config.prior.validate:
            prior.eval()
            validation_loss = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
                    loss = prior(image, caption, masks=mask)
                    validation_loss += loss.item()

            validation_loss = validation_loss / len(val_loader)

            if validation_loss <= best_loss:
                best_loss = validation_loss
                Path(config.prior.model_location).parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(prior.state_dict(), config.prior.model_location)
                print(
                    f"[Checkpoint] Prior best validation loss "
                    f"{best_loss:.5f}; saved to {config.prior.model_location}",
                    flush=True,
                )

            print(f"[Epoch {epoch + 1}/{config.prior.epochs}] Training Loss: {training_loss:.5f} | Validation Loss: {validation_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.3e} | Time: {perf_counter() - epoch_started:.1f}s", flush=True)
            
        else:
            Path(config.prior.model_location).parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(prior.state_dict(), config.prior.model_location)
            print(f"[Epoch {epoch + 1}/{config.prior.epochs}] Training Loss: {training_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.3e} | Time: {perf_counter() - epoch_started:.1f}s", flush=True)

    print(
        f"[Complete] Prior training finished. Checkpoint: "
        f"{config.prior.model_location}",
        flush=True,
    )
       
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the DALL-E 2 prior stage")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    if not config.using_pretrained_clip and not isfile(config.clip.model_location):
        print("CLIP model has not been trained. Training CLIP...")
        print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
        train_clip(config)
    
    print("Training Prior...")
    if config.using_pretrained_clip:
        print("Using frozen pretrained CLIP:", config.pretrained_clip_path)
    print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if config.device.type == "cuda" else "")
    print("Using dataset:", config.dataset, "from", config.data_location)
    train_prior(config)
