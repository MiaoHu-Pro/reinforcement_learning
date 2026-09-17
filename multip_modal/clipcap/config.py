import torch
from pathlib import Path


CLIPCAP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLIPCAP_DIR.parent.parent

# `python train.py`默认继续使用原来的2张图片/38条caption演示数据。
# `python train.py --dataset flickr8k`使用本地Flickr8k训练集。
DEFAULT_DATASET = "demo"
SUPPORTED_DATASETS = ("demo", "flickr8k")
DEMO_DATA_PATH = CLIPCAP_DIR / "caption_image.pkl"
FLICKR8K_DATA_PATH = PROJECT_ROOT / "datasets" / "flickr8k" / "data"

# Flickr8k共有6000张训练图片，每张图片有5条caption。第一次训练时，
# Chinese CLIP会为6000张图片各计算一次特征并缓存；以后运行直接读取缓存。
FLICKR8K_CAPTION_COLUMNS = tuple(f"caption_{index}" for index in range(5))
FLICKR8K_EMBED_CACHE_PATH = (
    CLIPCAP_DIR / "cache" / "flickr8k_train_chinese_clip_embeddings.pt"
)
CLIP_EMBED_BATCH_SIZE = 128

# demo非常小，保持原来的batch size；Flickr8k在A100上使用更大的batch，
# 避免30,000个image-caption pair产生过多小批次。
DEMO_BATCH_SIZE = 4
FLICKR8K_BATCH_SIZE = 64


# CLIP_MODEL_PATH = "~/scratch/llms_model/chinese-clip-vit-base-patch16"
CLIP_MODEL_PATH = Path(
    "~/scratch/llms_model/chinese-clip-vit-base-patch16"
).expanduser()

# 一张图片的嵌入经过投影转换成10个token的embedding，每个embedding的dim是768
IMAGE_TOKEN_LENGTH = 10  # 图片的token的数量
MAX_LENGTH = 50  # 最大token数量
# clip对接的大语言模型

# LLM_PATH = "~/scratch/llms_model/gpt2-chinese-cluecorpussmall"

LLM_PATH = Path(
    "~/scratch/llms_model/gpt2-chinese-cluecorpussmall"
).expanduser()

LLM_WORD_EMBD_DIM = 768  # gpt2的词嵌入维度
IMAGE_EMBD_DIM = 512  # clip输出的图像嵌入(特征向量)的维度
device = torch.device("cuda")
