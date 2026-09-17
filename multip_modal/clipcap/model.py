import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from typing import Sequence
from config import LLM_PATH, IMAGE_TOKEN_LENGTH, IMAGE_EMBD_DIM


class MLP(nn.Module):
    """投影层"""

    def __init__(self, sizes: Sequence[int]):
        super().__init__()

        in_dim, h1, out_dim = sizes
        self.l1 = nn.Linear(in_dim, h1)
        self.act1 = nn.Tanh()
        self.l2 = nn.Linear(h1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """512 ---> 3840 ---> 7680"""
        x = x.float()
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        return x


class ClipCaptionModel(nn.Module):
    def __init__(self):
        super(ClipCaptionModel, self).__init__()
        # 大语言模型：用来生成图片的文本描述
        self.gpt2 = AutoModelForCausalLM.from_pretrained(LLM_PATH)
        # gpt2的词嵌入维度是768
        self.word_embd_dim = self.gpt2.config.n_embd
        # 投影层定义
        self.projection = MLP((
            # 输入维度是512,也就是clip的图像编码器输出的嵌入的维度
            IMAGE_EMBD_DIM,
            # (768 * 10) // 2
            # 10是图片嵌入转换成的token数量
            (self.word_embd_dim * IMAGE_TOKEN_LENGTH) // 2,
            # 768 x 10，图片占10个token，每个token的嵌入和词嵌入相同是768维度
            self.word_embd_dim * IMAGE_TOKEN_LENGTH
        ))

    def forward(self, image_embeds, caption_ids, mask):
        # 张量形状：[B, 文本长度, gpt2的词嵌入的维度]
        # 标题caption的每个token的词嵌入
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        # 将图片的嵌入转换为像词嵌入那样的维度：
        # [B, 图片token的长度为10, gpt2的词嵌入的维度]
        image_as_word_embeds = self.projection(
            image_embeds
        ).view(-1, IMAGE_TOKEN_LENGTH, self.word_embd_dim)
        # 10个图片的token + 文本的token
        # 张量形状：[B, 10+文本长度, 词嵌入维度]
        embedding_cat = torch.cat((
            image_as_word_embeds,  # 图像的token
            caption_embeds  # 文本的token
        ), dim=1)
        out = self.gpt2(inputs_embeds=embedding_cat, attention_mask=mask)
        # 张量形状：[B, 10+文本长度，词嵌入维度]
        logits = out.logits
        return logits
