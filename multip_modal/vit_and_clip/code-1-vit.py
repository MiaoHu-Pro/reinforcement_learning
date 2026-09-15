import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model,    # 模型的维度
        img_size,   # 图片大小
        patch_size,  # 补丁大小
        n_channels  # 通道数量
    ):
        super().__init__()

        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.linear_project = nn.Conv2d(
            self.n_channels,  # in_channels
            self.d_model,  # out_channels
            kernel_size=self.patch_size,  # kernel_size
            stride=self.patch_size  # stride
        )

    # B: 批次大小
    # C: 通道数量
    # H: 图像高度
    # W: 图像宽度
    # P_col: 补丁的列
    # P_row: 补丁的行
    def forward(self, x):
        # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = self.linear_project(x)
        x = x.flatten(2)  # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2)  # (B, d_model, P) -> (B, P, d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # 类别token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 创建位置编码
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 为批次中的每张图片分配一个类别token
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        # 将类别token添加到每个图像的补丁嵌入数组的开头
        x = torch.cat((tokens_batch, x), dim=1)
        # 将位置编码添加到嵌入中
        x = x + self.pe
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # 计算Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q和K的点积
        attention = Q @ K.transpose(-2, -1)

        # 缩放
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_size) for _ in range(n_heads)
        ])

    def forward(self, x):
        # 拼接多个注意力头
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)

        # 多头注意力
        self.mha = MultiHeadAttention(d_model, n_heads)

        # 层归一化
        self.ln2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )

    def forward(self, x):
        # 第一次层归一化之后的残差
        out = x + self.mha(self.ln1(x))
        # 第二次层归一化之后的残差
        out = out + self.mlp(self.ln2(out))
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_classes,
        img_size,
        patch_size,
        n_channels,
        n_heads,
        n_layers
    ):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0  \
            and img_size[1] % patch_size[1] == 0, \
            "img_size 必须能被 patch_size 整除"
        assert d_model % n_heads == 0, \
            "d_model 必须能被 n_heads 整除"

        self.d_model = d_model  # 模型维度，嵌入的维度（宽度）
        self.n_classes = n_classes  # 类别的数量
        self.img_size = img_size  # 图片大小
        self.patch_size = patch_size  # 补丁大小
        self.n_channels = n_channels  # 通道数
        self.n_heads = n_heads  # 注意力头的数量
        # 补丁的数量 = (32x32) // (4x4)
        self.n_patches = (self.img_size[0] * self.img_size[1]) \
            // (self.patch_size[0] * self.patch_size[1])
        # 序列的长度 = 1（分类token） + 补丁的数量
        self.max_seq_length = self.n_patches + 1
        # 补丁嵌入
        self.patch_embedding = PatchEmbedding(
            self.d_model,
            self.img_size,
            self.patch_size,
            self.n_channels
        )
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            self.max_seq_length
        )
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(self.d_model, self.n_heads)
            for _ in range(n_layers)
        ])

        # 用于分类的MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # 将图片转换成补丁的嵌入（embedding）
        x = self.patch_embedding(images)
        # 添加位置编码
        x = self.positional_encoding(x)
        # 编码
        x = self.transformer_encoder(x)
        # 分类的线性层
        x = self.classifier(x[:, 0])
        return x


d_model = 9  # 嵌入的维度9
n_classes = 10  # 类别数量为10
img_size = (32, 32)  # 图片大小为32x32
patch_size = (16, 16)  # 补丁的大小是16x16
n_channels = 1  # 灰度图片通道数量为1
n_heads = 3  # 3个注意力头
n_layers = 3  # 3层编码器
batch_size = 128  # 每个批次128张图片
epochs = 10  # 训练5个epoch
alpha = 0.005  # 学习率5e-3


transform = T.Compose([
    T.Resize(img_size),  # 28x28 --> 32x32
    T.ToTensor()  # 转换成torch.tensor
])

train_set = MNIST(
    root="./datasets", train=True, download=True, transform=transform
)
test_set = MNIST(
    root="./datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

device = torch.device("cuda")

ViT = VisionTransformer(
    d_model,
    n_classes,
    img_size,
    patch_size,
    n_channels,
    n_heads,
    n_layers
).to(device)

optimizer = Adam(ViT.parameters(), lr=alpha)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    training_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 取出图像和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # 前向传播
        outputs = ViT(inputs)
        # 交叉熵损失
        loss = criterion(outputs, labels)
        # 求导数
        loss.backward()
        # 梯度下降
        optimizer.step()

        training_loss += loss.item()

    print(
        f'Epoch {epoch + 1}/{epochs} loss: {training_loss / len(train_loader):.3f}')


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = ViT(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'\n预测准确率: {100 * correct // total} %')
