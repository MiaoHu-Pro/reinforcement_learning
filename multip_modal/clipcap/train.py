import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from clipcap_dataset import ClipCapDataset
from model import ClipCaptionModel
import torch.nn.functional as F
from config import (
    DEFAULT_DATASET,
    DEMO_BATCH_SIZE,
    FLICKR8K_BATCH_SIZE,
    IMAGE_TOKEN_LENGTH,
    LLM_PATH,
    SUPPORTED_DATASETS,
    device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练ClipCap图片描述模型")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=DEFAULT_DATASET,
        help="默认demo使用2张演示图片；flickr8k使用本地Flickr8k train split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="不设置时demo使用4，Flickr8k使用64",
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size必须大于0")
    if args.epochs < 1:
        parser.error("--epochs必须大于0")
    return args


def train(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for data in progress:
            image_embed, caption_ids, mask = data
            image_embed = image_embed.to(device)
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            # 输出的logits
            logits = model(image_embed, caption_ids, mask)

            # 计算loss
            # [图片的最后一个token]，[两]，[只]，[狗]
            #          ↓            ↓    ↓
            #         [两]         [只]  [狗]
            shift_logits = logits[
                :,
                # 截取范围[图片的最后一个token～倒数第二个token]
                IMAGE_TOKEN_LENGTH - 1: -1,  # 去掉最后一个token
                :
            ].contiguous().view(-1, logits.size(-1))
            # 预测目标
            # mask的前IMAGE_TOKEN_LENGTH个位置属于图片prefix；其余位置与
            # caption_ids一一对应。padding位置不能参与loss，否则大量PAD
            # token会主导训练，使模型倾向于生成PAD而不是学习图片描述。
            caption_mask = mask[:, IMAGE_TOKEN_LENGTH:]
            shift_labels = caption_ids.masked_fill(
                caption_mask == 0,
                -100,  # CrossEntropyLoss默认忽略的label
            ).contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
            )
            # logits.size(-1): 取最后一维词表大小vocab_size。
            # 原logits形状是[B, IMAGE_TOKEN_LENGTH + caption_length, V]。
            # 切片后shift_logits对应每一个caption目标token。
            # 再 `.contiguous().view(-1, logits.size(-1))`
            # 把batch和序列维展平，便于和shift_labels计算交叉熵。

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), f'model.pt')


def main():
    args = parse_args()
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)

    # Flickr8k第一次运行会先用Chinese CLIP计算并缓存图片特征。先构造
    # dataset、再加载GPT-2，可避免两个预训练模型同时占用GPU显存。
    dataset = ClipCapDataset(tokenizer, dataset_name=args.dataset)
    default_batch_size = (
        DEMO_BATCH_SIZE
        if args.dataset == "demo"
        else FLICKR8K_BATCH_SIZE
    )
    batch_size = args.batch_size or default_batch_size
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    # 加载模型
    model = ClipCaptionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f"训练batch size: {batch_size}")
    print(f"训练epoch数: {args.epochs}")
    train(model, train_dataloader, optimizer, args.epochs)


if __name__ == '__main__':
    main()
