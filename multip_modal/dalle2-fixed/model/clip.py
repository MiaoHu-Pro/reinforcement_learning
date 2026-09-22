import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.transformer import PatchEmbedding, TransformerBlock


class TextEncoder(nn.Module):
    """clip文本编码器结构"""

    def __init__(self, config):
        super().__init__()

        # Text embedding table
        self.encoder_embedding = nn.Embedding(
            config.vocab_size, config.clip.text_width)

        # Standard deviation for initializing parameters
        param_std = config.clip.text_width ** -0.5

        # Learned positional encodings
        self.positional_encodings = nn.Parameter(
            param_std * torch.randn(config.text_seq_length, config.clip.text_width))

        # Dropout
        self.dropout = nn.Dropout(config.clip.dropout)

        # Transformer encoder blocks
        self.encoder = nn.ModuleList(
            [TransformerBlock(
                config.clip.text_width,
                cond_width=config.clip.text_width,
                n_heads=config.clip.text_heads,
                dropout=config.clip.dropout,
                r_mlp=config.clip.r_mlp,
                bias=config.clip.bias
            ) for _ in range(config.clip.text_layers)]
        )

        # Final layer normalization
        self.final_ln = nn.LayerNorm(config.clip.text_width)

        # Learned projection of text to latent space
        self.projection = nn.Parameter(
            param_std * torch.randn(config.clip.text_width, config.latent_dim))

    def forward(self, text, mask=None, get_all_features=False):
        # Get text embeddings
        # (B, text_seq_length) -> # (B, text_seq_length, text_width)
        x = self.encoder_embedding(text)

        # Add positional encodings
        x = x + self.positional_encodings

        # Dropout
        x = self.dropout(x)

        # Pass through transformer encoder blocks
        for block in self.encoder:
            x = block(x, mask=mask)

        # Apply final layer normalization
        x = self.final_ln(x)

        if get_all_features:
            return x

        # Take features from the EOT embedding
        # (B, text_seq_length, text_width) -> (B, text_width)
        x = x[torch.arange(text.shape[0]), torch.sub(
            torch.sum(mask, dim=1), 1)]

        # Joint multimodal embedding
        x = x @ self.projection  # (B, text_width) -> (B, latent_dim)

        return x


class ImageEncoder(nn.Module):
    """图像编码器"""
    def __init__(self, config):
        super().__init__()

        assert (config.img_size[0] % config.clip.patch_size[0] == 0) and (
            config.img_size[1] % config.clip.patch_size[1] == 0), "img_size dimensions must be divisible by patch_size dimensions"

        # Calculating number of patches based on image and patch sizes
        n_patches = (config.img_size[0] * config.img_size[1]) // (
            config.clip.patch_size[0] * config.clip.patch_size[1])

        # Length equal to number of patches plus 1 for the classification token
        vit_seq_length = n_patches + 1

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            config.img_channels,
            config.clip.vit_width,
            config.clip.patch_size
        )

        # Standard deviation for initializing parameters
        param_std = config.clip.vit_width ** -0.5

        # Classification token
        self.cls_token = nn.Parameter(
            param_std * torch.randn(1, 1, config.clip.vit_width))

        # Learned positional encodings
        self.positional_encodings = nn.Parameter(
            param_std * torch.randn(vit_seq_length, config.clip.vit_width))

        # Dropout
        self.dropout = nn.Dropout(config.clip.dropout)

        # Layer normalization before transformer
        self.pre_ln = nn.LayerNorm(config.clip.vit_width)

        # Transformer encoder blocks
        self.encoder = nn.ModuleList(
            [TransformerBlock(
                config.clip.vit_width,
                cond_width=config.clip.vit_width,
                n_heads=config.clip.vit_heads,
                dropout=config.clip.dropout,
                r_mlp=config.clip.r_mlp,
                bias=config.clip.bias
            ) for _ in range(config.clip.vit_layers)]
        )

        # Final layer normalization
        self.final_ln = nn.LayerNorm(config.clip.vit_width)

        # Learned projection of image to latent space
        self.projection = nn.Parameter(
            param_std * torch.randn(config.clip.vit_width, config.latent_dim))

    def forward(self, x, get_all_features=False):
        # Get patch embeddings
        # (B, C, H, W) -> (B, n_patches, vit_width)
        x = self.patch_embedding(x)

        # Add class tokens to patches
        # (B, n_patches, vit_width) -> (B, vit_seq_length, vit_width)
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1), x), dim=1)

        # Add positional encodings
        x = x + self.positional_encodings

        # Dropout
        x = self.dropout(x)

        # Apply layer normalization before transformer
        x = self.pre_ln(x)

        # Pass through transformer encoder blocks
        for block in self.encoder:
            x = block(x)

        # Apply final layer normalization
        x = self.final_ln(x)

        if get_all_features:
            return x

        # Take class tokens
        x = x[:, 0, :]  # (B, vit_seq_length, vit_width) -> (B, vit_width)

        # Joint multimodal embedding
        x = x @ self.projection  # (B, vit_width) -> (B, latent_dim)

        return x


class CLIP(nn.Module):
    """clip模型的结构"""
    def __init__(self, config):
        super().__init__()

        # Vision transformer
        self.image_encoder = ImageEncoder(config)

        # Text transformer
        self.text_encoder = TextEncoder(config)

        # Learned temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text, mask=None):
        # Get text and image features
        I_e = self.image_encoder(image)  # (B, C, H, W) -> (B, latent_dim)
        # (B, text_seq_length) -> (B, latent_dim)
        T_e = self.text_encoder(text, mask=mask)

        I_e = nn.functional.normalize(I_e, dim=-1)
        T_e = nn.functional.normalize(T_e, dim=-1)

        # Scaled pairwise cosine similarities
        # CLIP stores inverse temperature in log space. Capping at 100 follows
        # the original CLIP implementation and avoids exponential overflow.
        logit_scale = self.temperature.exp().clamp(max=100.0)
        logits = (I_e @ T_e.transpose(-2, -1)) * logit_scale

        # Symmetric loss function
        labels = torch.arange(logits.shape[0], device=image.device)

        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2

        return loss


class PretrainedCLIPAdapter(nn.Module):
    """Frozen Hugging Face CLIP with the interface used by this project.

    Dataset images arrive in this project's normalized 64x64 representation.
    The adapter converts them back to [0, 1], resizes to the pretrained CLIP
    resolution, and applies CLIP's own channel statistics. Text is decoded from
    the project's byte tokens and retokenized by the pretrained CLIP tokenizer.
    """

    def __init__(self, config):
        super().__init__()
        from transformers import AutoTokenizer, CLIPModel

        self.model = CLIPModel.from_pretrained(
            config.pretrained_clip_path,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_clip_path,
            local_files_only=True,
        )
        projection_dim = int(self.model.config.projection_dim)
        if projection_dim != config.latent_dim:
            raise ValueError(
                "Pretrained CLIP projection dimension does not match "
                f"config.latent_dim: {projection_dim} != {config.latent_dim}"
            )

        self.image_size = int(self.model.config.vision_config.image_size)
        self.register_buffer(
            "dataset_mean",
            torch.tensor(config.train_mean).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dataset_std",
            torch.tensor(config.train_std).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )

    @staticmethod
    def _feature_tensor(output):
        """Support both Transformers 4.x tensors and 5.x model outputs."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError(f"Unsupported CLIP feature output: {type(output)!r}")

    @staticmethod
    def _decode_byte_text(captions, masks):
        texts = []
        captions_cpu = captions.detach().cpu()
        masks_cpu = masks.detach().cpu()
        for token_ids, mask in zip(captions_cpu, masks_cpu):
            valid_length = int(mask.sum().item())
            content = bytes(
                int(token_id)
                for token_id in token_ids[1:max(1, valid_length - 1)]
            )
            texts.append(content.decode("utf-8", errors="replace"))
        return texts

    def image_encoder(self, images):
        pixels = images * self.dataset_std + self.dataset_mean
        if pixels.shape[1] == 1:
            pixels = pixels.repeat(1, 3, 1, 1)
        if pixels.shape[1] != 3:
            raise ValueError("Pretrained CLIP expects one or three image channels")
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        pixels = (pixels - self.clip_mean) / self.clip_std
        features = self._feature_tensor(
            self.model.get_image_features(pixel_values=pixels)
        )
        return F.normalize(features.float(), dim=-1)

    def text_encoder(self, captions, mask=None):
        if mask is None:
            raise ValueError("A byte-token attention mask is required")
        texts = self._decode_byte_text(captions, mask)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).to(captions.device)
        features = self._feature_tensor(self.model.get_text_features(**inputs))
        return F.normalize(features.float(), dim=-1)


def build_clip_encoder(config):
    """Build either the local pretrained CLIP or the custom trained CLIP."""
    if config.using_pretrained_clip:
        return PretrainedCLIPAdapter(config).to(config.device)

    clip = CLIP(config).to(config.device)
    clip.load_state_dict(
        torch.load(
            config.clip.model_location,
            map_location=config.device,
            weights_only=True,
        )
    )
    return clip
