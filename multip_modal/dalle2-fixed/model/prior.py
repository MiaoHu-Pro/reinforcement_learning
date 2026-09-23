"""Diffusion prior: CLIP text embedding -> CLIP image embedding."""

import torch
import torch.nn as nn

from data.data_utils import (
    extract_and_expand,
    forward_diffusion,
    freeze_model,
    get_schedule_values,
)
from model.clip import build_clip_encoder
from model.transformer import SinusoidalPositionalEmbedding, TransformerBlock


class DiffusionPrior(nn.Module):
    """Predict a clean CLIP image embedding from its noisy version and text.

    Training samples a random diffusion time and predicts x_0. Sampling starts
    from Gaussian noise and repeatedly applies the corresponding DDPM posterior
    update. The old implementation made only one prediction at t=T, which was
    not a reverse-diffusion process.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.clip = build_clip_encoder(config)
        freeze_model(self.clip)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(
                config.prior.max_time,
                config.latent_dim,
            ),
            nn.Linear(
                config.latent_dim,
                config.latent_dim * config.prior.r_mlp,
                bias=config.prior.bias,
            ),
            nn.SiLU(),
            nn.Linear(
                config.latent_dim * config.prior.r_mlp,
                config.latent_dim,
                bias=config.prior.bias,
            ),
        )
        self.learned_embedding = nn.Parameter(torch.randn(config.latent_dim))
        self.schedule_values = get_schedule_values(
            schedule=config.prior.schedule,
            max_time=config.prior.max_time,
            device=config.device,
        )

        # Keep the historical attribute name so existing FashionMNIST
        # checkpoints remain loadable after the sampling fixes.
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    config.latent_dim,
                    cond_width=config.latent_dim,
                    n_heads=config.prior.n_heads,
                    dropout=config.prior.dropout,
                    r_mlp=config.prior.r_mlp,
                    bias=config.prior.bias,
                )
                for _ in range(config.prior.n_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(
                config.latent_dim,
                config.latent_dim,
                bias=config.prior.bias,
            ),
        )
        self.register_buffer(
            "causal_attention_mask",
            torch.tril(torch.ones(5, 5))[None, :],
        )

    def train(self, mode=True):
        """Keep the frozen CLIP submodel deterministic while training prior."""
        super().train(mode)
        self.clip.eval()
        return self

    def _caption_token(self, captions):
        """Pad/truncate byte token IDs into one latent-width token."""
        if captions.shape[1] >= self.config.latent_dim:
            captions = captions[:, : self.config.latent_dim]
        else:
            captions = nn.functional.pad(
                captions,
                (0, self.config.latent_dim - captions.shape[1]),
            )
        # Raw byte IDs span [0, 255]. Scaling prevents this token from
        # overwhelming the CLIP and timestep embeddings numerically.
        captions = captions.to(dtype=self.learned_embedding.dtype) / 255.0
        return captions[:, None, :]

    def _predict_x0(
        self,
        noisy_image_embeddings,
        timesteps,
        text_embeddings,
        caption_tokens,
    ):
        timestep_embeddings = self.time_mlp(timesteps)[:, None, :]
        learned_embeddings = self.learned_embedding.expand(
            noisy_image_embeddings.shape[0], -1
        )[:, None, :]
        tokens = torch.cat(
            (
                caption_tokens,
                text_embeddings,
                timestep_embeddings,
                noisy_image_embeddings[:, None, :],
                learned_embeddings,
            ),
            dim=1,
        )
        for block in self.decoder:
            tokens = block(tokens, mask=self.causal_attention_mask)
        return self.output(tokens[:, -1, :])

    def _one_reverse_trajectory(self, text_embeddings, caption_tokens):
        batch_size = text_embeddings.shape[0]
        x_t = torch.randn(
            batch_size,
            self.config.latent_dim,
            device=text_embeddings.device,
        )
        schedule = self.schedule_values

        for time_index in reversed(range(self.config.prior.max_time)):
            timesteps = torch.full(
                (batch_size,),
                time_index,
                device=text_embeddings.device,
                dtype=torch.long,
            )
            pred_x0 = self._predict_x0(
                x_t,
                timesteps,
                text_embeddings,
                caption_tokens,
            )
            if self.config.using_pretrained_clip:
                # The adapter's training targets are unit-normalized CLIP
                # embeddings. Keep reverse diffusion on that learned support.
                pred_x0 = nn.functional.normalize(pred_x0, dim=-1)

            alpha_t = extract_and_expand(
                schedule["alphas"], timesteps, x_t.shape
            )
            beta_t = extract_and_expand(
                schedule["betas"], timesteps, x_t.shape
            )
            alpha_bar_t = extract_and_expand(
                schedule["alpha_bars"], timesteps, x_t.shape
            )
            alpha_bar_prev = extract_and_expand(
                schedule["alpha_bars_prev"], timesteps, x_t.shape
            )
            denominator = (1.0 - alpha_bar_t).clamp_min(1e-12)
            posterior_mean = (
                beta_t * alpha_bar_prev.sqrt() / denominator * pred_x0
                + (1.0 - alpha_bar_prev)
                * alpha_t.sqrt()
                / denominator
                * x_t
            )
            if time_index > 0:
                sigma_t = extract_and_expand(
                    schedule["sigma"], timesteps, x_t.shape
                )
                x_t = posterior_mean + sigma_t * torch.randn_like(x_t)
            else:
                x_t = posterior_mean
        return x_t

    @torch.no_grad()
    def sample(self, captions, masks=None, num_candidates=2):
        """Generate image embeddings and rerank candidates with CLIP text."""
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        text_embedding = self.clip.text_encoder(captions, mask=masks)
        text_tokens = text_embedding[:, None, :]
        caption_tokens = self._caption_token(captions)
        candidates = torch.stack(
            [
                self._one_reverse_trajectory(text_tokens, caption_tokens)
                for _ in range(num_candidates)
            ],
            dim=1,
        )
        scores = torch.nn.functional.cosine_similarity(
            candidates,
            text_embedding[:, None, :],
            dim=-1,
        )
        best = scores.argmax(dim=1)
        batch_indices = torch.arange(captions.shape[0], device=captions.device)
        return candidates[batch_indices, best]

    def forward(self, images, captions, masks=None):
        """Denoising objective for a random embedding diffusion timestep."""
        with torch.no_grad():
            image_embeddings = self.clip.image_encoder(images)
            text_embeddings = self.clip.text_encoder(captions, mask=masks)

        timesteps = torch.randint(
            0,
            self.config.prior.max_time,
            (images.shape[0],),
            device=images.device,
            dtype=torch.long,
        )
        noisy_embeddings, _ = forward_diffusion(
            image_embeddings,
            self.schedule_values,
            timesteps,
        )
        pred_image_embeddings = self._predict_x0(
            noisy_embeddings,
            timesteps,
            text_embeddings[:, None, :],
            self._caption_token(captions),
        )
        return nn.functional.mse_loss(
            pred_image_embeddings,
            image_embeddings,
        )
