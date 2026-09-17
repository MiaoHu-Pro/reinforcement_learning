import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from clipcap_dataset import ClipCapDataset
from model import ClipCaptionModel
import torch.nn.functional as F
from config import LLM_PATH, IMAGE_TOKEN_LENGTH, device


def train(model, train_loader, optimizer):
    model.train()
    for _ in range(20):
        for _, data in enumerate(tqdm(train_loader)):
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

    torch.save(model.state_dict(), f'model.pt')


def main():
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    # 加载模型
    model = ClipCaptionModel().to(device)

    dataset = ClipCapDataset(tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, train_dataloader, optimizer)


if __name__ == '__main__':
    main()
