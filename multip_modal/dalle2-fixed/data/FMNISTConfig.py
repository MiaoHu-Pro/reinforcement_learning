from dataclasses import dataclass, field
from pathlib import Path

import torch


DALLE2_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DALLE2_DIR.parent.parent

@dataclass
class CLIPConfig:
    # Vision Transformer
    # CLIP图片编码器的相关配置
    patch_size:tuple[int,int] = (4,4) # 补丁大小4x4
    vit_width:int = 256 # 图片补丁嵌入的维度
    vit_layers:int = 6
    vit_heads:int = 8
    # Text Transformer
    # CLIP文本编码器
    text_width:int = 256 # 词嵌入的维度
    text_layers:int = 6
    text_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = True
    validate:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    get_val_accuracy:bool = False
    model_location:str = "./trained_models/clip_fmnist.pt"

@dataclass
class PriorConfig:
    """先验模型的配置"""
    # Diffusion
    max_time:int = 1000 # 给clip图片的特征向量加噪声的最大时间步
    schedule:str = "cosine" # 使用余弦方差调度计划
    schedule_offset:float = 0.008
    # Transformer Decoder 仅解码器架构的transformer
    width:int = 256 # 先验模型输入的5个token，每个token的维度
    n_layers:int = 6
    n_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = False
    validate:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 150
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    model_location:str = "./trained_models/prior_fmnist.pt"

@dataclass
class DecoderConfig:
    """条件扩散模型"""
    # Diffusion
    max_time:int = 1000 # 给图片加噪声的最大时间步
    schedule:str = "cosine"
    # UNet
    n_groups:int = 8 # 组归一化中一组有8条数据
    kernel_size:tuple[int, int] = (3,3)
    model_channels:int = 32
    cond_channels:int = 128
    channel_ratios:list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layer_blocks:int = 2
    dropout:float = 0.1
    use_scale_shift:bool = True
    n_heads:int = 8
    stride:int = 2
    down_pool:bool = False
    r_mlp:int = 4
    bias:bool = False
    text_layers:int = 4
    n_img_tokens:int = 4 # 图片特征向量要作为条件注入给注意力模块，需要转换成4个token
    # Training
    augment_data:bool = False
    validate:bool = False
    num_workers:int = 0
    batch_size:int = 32
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 100
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    sample_after_epoch:bool = False
    model_location:str = "./trained_models/decoder_fmnist.pt"

@dataclass
class FMNISTConfig:
    latent_dim:int = 256
    # Dataset Info
    dataset:str = "fashion_mnist"
    data_location:str = str(DALLE2_DIR / "datasets")
    img_size:tuple[int,int] = (32,32)
    img_channels:int = 1
    vocab_size:int = 256 # 我们使用ascii码进行分词，所以词汇表大小256
    text_seq_length:int = 64 # 提示词最大序列长度为64
    # Data Augmentation / Normalization
    # 对图片进行数据增强/归一化的配置
    prob_hflip:float = 0.5
    crop_padding:int = 4
    train_mean:list[float] = field(default_factory=lambda: [0.2855552])
    train_std:list[float] = field(default_factory=lambda: [0.33848408])
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    device:torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    # Model Configs
    clip:CLIPConfig = field(default_factory=CLIPConfig)
    prior:PriorConfig = field(default_factory=PriorConfig)
    decoder:DecoderConfig = field(default_factory=DecoderConfig)


def configure_dataset(
    config: FMNISTConfig,
    dataset: str,
    data_location: str | Path | None = None,
) -> FMNISTConfig:
    """Apply every shape/path setting that must agree for one dataset.

    A checkpoint trained for FashionMNIST cannot be loaded into the RGB
    Flickr8k architecture, so each dataset gets separate checkpoint names.
    """
    if dataset == "fashion_mnist":
        config.dataset = dataset
        config.data_location = str(
            Path(data_location).expanduser()
            if data_location is not None
            else DALLE2_DIR / "datasets"
        )
        config.img_size = (32, 32)
        config.img_channels = 1
        config.train_mean = [0.2855552]
        config.train_std = [0.33848408]
        config.train_val_split = (50000, 10000)
        # Corrected prior/decoder semantics are not compatible with weights
        # trained by the copied implementation. Keep those files untouched and
        # write the corrected pipeline to distinct checkpoint names.
        suffix = "fmnist_fixed"
    elif dataset == "flickr8k":
        config.dataset = dataset
        config.data_location = str(
            Path(data_location).expanduser()
            if data_location is not None
            else PROJECT_ROOT / "datasets" / "flickr8k" / "data"
        )
        # 64x64 keeps this from-scratch demo tractable. It is not the image
        # resolution used by the production DALL-E 2 system.
        config.img_size = (64, 64)
        config.img_channels = 3
        config.train_mean = [0.5, 0.5, 0.5]
        config.train_std = [0.5, 0.5, 0.5]
        config.train_val_split = (6000, 1000)
        suffix = "flickr8k"
    else:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; choose fashion_mnist or flickr8k"
        )

    model_dir = DALLE2_DIR / "trained_models"
    config.clip.model_location = str(model_dir / f"clip_{suffix}.pt")
    config.prior.model_location = str(model_dir / f"prior_{suffix}.pt")
    config.decoder.model_location = str(model_dir / f"decoder_{suffix}.pt")
    return config
