import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        super().__init__()
        # 创建一个 (token的数量, 嵌入的维度) 形状的全0张量
        # width 就是 d_model
        pe = torch.zeros(max_seq_length, width)
        # 将位置编码信息填充到pe中
        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/width)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/width)))
        # 位置编码信息进行冻结，不参与反向传播
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 直接将位置编码和token的嵌入进行相加
        x = x + self.pe
        return x


class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x, mask=None):
        # 计算K，Q，V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q和K的点积
        attention = Q @ K.transpose(-2, -1)
        # 缩放
        attention = attention / (self.head_size ** 0.5)
        # 掩码
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()
        self.head_size = width // n_heads
        self.W_o = nn.Linear(width, width)
        self.heads = nn.ModuleList([
            AttentionHead(width, self.head_size) for _ in range(n_heads)
        ])

    def forward(self, x, mask=None):
        # 拼接多个注意力头
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4):
        super().__init__()
        self.width = width  # 嵌入的维度d_model
        self.n_heads = n_heads

        # 层归一化
        self.ln1 = nn.LayerNorm(width)

        # 多头注意力
        self.mha = MultiHeadAttention(width, n_heads)

        # 层归一化
        self.ln2 = nn.LayerNorm(width)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width*r_mlp),
            nn.GELU(),
            nn.Linear(self.width*r_mlp, self.width)
        )

    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3)  # 添加 SOT token 和 EOT token
        out = out + "".join([
            chr(0) for _ in range(max_seq_length-len(out))
        ])  # 添加Padding
        out = torch.IntTensor(list(out.encode("utf-8")))  # 对文本进行编码
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((
            mask,
            torch.zeros(max_seq_length - len(mask))
        )).type(torch.IntTensor)
    else:
        # 将input_ids解码为text文本
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,  # 词汇表大小=256
        width,  # 宽度d_model
        max_seq_length,  # 文本最大长度
        n_heads,
        n_layers,
        emb_dim  # 嵌入维度
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = PositionalEmbedding(
            width,
            max_seq_length
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(width, n_heads)
            for _ in range(n_layers)
        ])
        # 可学习投影（projection）
        # d_model(width) --- emb_dim
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        # 文本嵌入
        x = self.encoder_embedding(text)
        # 位置嵌入
        x = self.positional_embedding(x)
        # Transformer编码器
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)
        # 从EOT的嵌入抽取特征
        x = x[
            torch.arange(text.shape[0]),  # 批次中数据的索引
            # 取出掩码mask矩阵的第0行，加和再减1，就得到了EOT的索引
            torch.sub(torch.sum(mask[:, 0], dim=1), 1)
        ]
        # 将文本特征嵌入到联合嵌入空间中（多模态嵌入空间）
        # 文本编码器输出的张量的维度和图像编码器输出的张量的维度必须一致
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        width,  # 补丁嵌入的维度d_model
        img_size,
        patch_size,
        n_channels,
        n_layers,
        n_heads,
        emb_dim
    ):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0  \
            and img_size[1] % patch_size[1] == 0, \
            "img_size必须能被patch_size整除"
        assert width % n_heads == 0, \
            "width必须能被n_heads整除"

        self.n_patches = (img_size[0] * img_size[1]) \
            // (patch_size[0] * patch_size[1])
        self.max_seq_length = self.n_patches + 1
        self.linear_project = nn.Conv2d(
            n_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positional_embedding = PositionalEmbedding(
            width,
            self.max_seq_length
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(width, n_heads)
            for _ in range(n_layers)
        ])

        # 可学习的投影
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, x):
        # 补丁嵌入
        x = self.linear_project(x)
        x = x.flatten(2).transpose(1, 2)

        # 位置嵌入
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1), x), dim=1)
        x = self.positional_embedding(x)

        # Transformer编码器
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # 获取类别token
        x = x[:, 0, :]

        # 多模态嵌入
        # 保证文本编码器的输出的维度和图像编码器的输出的维度相等
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim,  # 经过可学习投影后的嵌入维度
        vit_width,  # 图像编码器的宽度(d_model)
        img_size,
        patch_size,
        n_channels,
        vit_layers,
        vit_heads,
        vocab_size,
        text_width,  # 文本编码器的宽度（d_model）
        max_seq_length,
        text_heads,
        text_layers
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            vit_width,
            img_size,
            patch_size,
            n_channels,
            vit_layers,
            vit_heads,
            emb_dim
        )
        self.text_encoder = TextEncoder(
            vocab_size,
            text_width,
            max_seq_length,
            text_heads,
            text_layers,
            emb_dim
        )
        # 可学习温度
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device("cuda")

    def forward(self, image, text, mask=None):
        # Iₑ是图像嵌入，形状 [B, D=emb_dim]
        I_e = self.image_encoder(image)
        # Tₑ是文本嵌入，形状 [B, D=emb_dim]
        T_e = self.text_encoder(text, mask=mask)

        # 缩放逐点余弦相似度[n, n]
        # 形状 I_e @ T_e^T : [B, D] @ [D, B] --> [B, B]
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

        # 对称损失函数 labels形状为[B]，值为 [0, 1, 2, ..., B-1]
        labels = torch.arange(logits.shape[0]).to(self.device)
        # 从文本 --> 图像方向，以文本嵌入 T₃ 为例子，
        # 交叉熵损失的目标是让 T₃⋅I₃ 越大越好
        loss_i = nn.functional.cross_entropy(
            logits.transpose(-2, -1),
            labels
        )
        # 从图像 --> 文本方向，以图像嵌入 I₃ 为例子，
        # 交叉熵损失的目标是让 I₃⋅T₃ 越大越好
        loss_t = nn.functional.cross_entropy(
            logits,
            labels
        )
        # 两个方向的损失求平均值
        loss = (loss_i + loss_t) / 2

        return loss


class MNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("./clip-mnist/")
        self.transform = T.ToTensor()
        if train:
            self.split = "train"
        else:
            self.split = "test"

        self.captions = {
            0: "An image of 0",
            1: "An image of 1",
            2: "An image of 2",
            3: "An image of 3",
            4: "An image of 4",
            5: "An image of 5",
            6: "An image of 6",
            7: "An image of 7",
            8: "An image of 8",
            9: "An image of 9"
        }

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self, i):
        # 取出第i张图片
        img = self.dataset[self.split][i]["image"]
        # 转换成张量
        img = self.transform(img)
        # 图片对应的文本，以及掩码
        cap, mask = tokenizer(
            self.captions[self.dataset[self.split][i]["label"]]
        )
        # 为什么要repeat？
        mask = mask.repeat(len(mask), 1)
        return {"image": img, "caption": cap, "mask": mask}


emb_dim = 32  # 文本编码器和图像编码器输出的张量的维度
vit_width = 9  # 图像编码器的嵌入的宽度
img_size = (28, 28)
patch_size = (14, 14)
n_channels = 1
vit_layers = 3  # 图像编码器中编码器的层的数量
vit_heads = 3  # 图像编码器中注意力头的数量
vocab_size = 256  # 词汇表大小
text_width = 32  # 文本编码器的嵌入的宽度
max_seq_length = 32  # 最大序列长度
text_heads = 8  # 文本编码器中注意力头的数量
text_layers = 4  # 文本编码器中编码器的层数
lr = 1e-3  # 学习率
epochs = 10
batch_size = 128


train_set = MNIST(train=True)
test_set = MNIST(train=False)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


device = torch.device("cuda")

model = CLIP(
    emb_dim,
    vit_width,
    img_size,
    patch_size,
    n_channels,
    vit_layers,
    vit_heads,
    vocab_size,
    text_width,
    max_seq_length,
    text_heads,
    text_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

best_loss = np.inf
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        img = data["image"].to(device)
        cap = data["caption"].to(device)
        mask = data["mask"].to(device)
        loss = model(img, cap, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")

    # 保存模型
    if loss.item() <= best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "./clip.pt")
        print("模型已经保存.")


# 加载最好的模型
model = CLIP(
    emb_dim,
    vit_width,
    img_size,
    patch_size,
    n_channels,
    vit_layers,
    vit_heads,
    vocab_size,
    text_width,
    max_seq_length,
    text_heads,
    text_layers
).to(device)
model.load_state_dict(torch.load("./clip.pt", map_location=device))

# 获取数据集的标签和图片进行对比
text = torch.stack(
    [tokenizer(x)[0] for x in test_set.captions.values()]
).to(device)
mask = torch.stack(
    [tokenizer(x)[1] for x in test_set.captions.values()]
)
mask = mask.repeat(
    1,
    len(mask[0])
).reshape(
    len(mask),
    len(mask[0]
        ), len(mask[0])).to(device)

correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        # 图像
        images = data["image"].to(device)
        # 文本
        labels = data["caption"].to(device)
        # 使用clip模型中的图像编码器对图像抽取特征
        image_features = model.image_encoder(images)
        # 使用clip模型中的文本编码器对文本抽取特征
        text_features = model.text_encoder(text, mask=mask)
        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # I_e @ T_e^T
        similarity = (
            100.0 * image_features @ text_features.T
        ).softmax(dim=-1)
        _, indices = torch.max(similarity, 1)
        # 预测结果
        pred = torch.stack([
            tokenizer(test_set.captions[int(i)])[0]
            for i in indices
        ]).to(device)
        # 预测正确的样本数量
        correct += int(sum(torch.sum((pred == labels), dim=1)//len(pred[0])))
        total += len(labels)

print(f'\n预测准确率: {100 * correct // total} %')
