import torch
from pathlib import Path



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
