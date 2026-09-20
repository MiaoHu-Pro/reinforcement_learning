import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码"""
    def __init__(self,
                 max_seq_length,    # Maximum sequence length
                 width              # Width of model
            ):
        super().__init__()

        # Create positional encodings
        pe = torch.zeros(max_seq_length, width)

        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos / (10000 ** (i / width)))
                else:
                    pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / width)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get positional encodings corresponding to inputted timesteps
        x = self.pe[x]

        return x

class PatchEmbedding(nn.Module):
    """图像补丁嵌入类"""
    def __init__(self,
                 img_channels, # Number of image channels
                 model_width,  # Width of model
                 patch_size    # Size of patches
            ):
        super().__init__()

        self.linear_project = nn.Conv2d(img_channels, model_width, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, model_width, P_r, P_c)

        x = x.flatten(2) # (B, model_width, P_r, P_c) -> (B, model_width, n_patches)

        x = x.transpose(1, 2) # (B, model_width, n_patches) -> (B, n_patches, model_width)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_kv,          # Width of model
                 d_q,           # Width of condition input
                 n_heads=1,     # Number of attention heads
                 dropout=0.,    # Dropout rate
                 bias=False     # Linear layer bias
            ):
        super().__init__()

        assert d_kv % n_heads == 0, "model_width must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_size = d_q // n_heads
        self.scale = self.head_size ** -0.5

        self.query = nn.Linear(d_q, d_q, bias=bias)
        self.key = nn.Linear(d_kv, d_q, bias=bias)
        self.value = nn.Linear(d_kv, d_q, bias=bias)

        self.out_proj = nn.Linear(d_kv, d_kv, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cond=None):
        if cond is None:
            cond = x

        # Obtain query heads
        Q = self.query(x) # (B, seq_len, model_width) -> (B, seq_len, model_width)
        Q = Q.view(x.shape[0], x.shape[1], self.n_heads, self.head_size) # (B, seq_len, model_width) -> (B, seq_len, n_heads, head_size)
        Q = Q.transpose(1, 2)  # (B, seq_len, n_heads, head_size) -> (B, n_heads, seq_len, head_size)

        # Obtain key heads
        K = self.key(cond) # (B, seq_len, cond_width) -> (B, seq_len, model_width)
        K = K.view(cond.shape[0], cond.shape[1], self.n_heads, self.head_size)
        K = K.transpose(1, 2)

        # Obtain value heads
        V = self.value(cond)  # (B, seq_len, cond_width) -> (B, seq_len, model_width)
        V = V.view(cond.shape[0], cond.shape[1], self.n_heads, self.head_size)
        V = V.transpose(1, 2)

        # Get dot product between queries and keys
        attention = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, seq_len, head_size) @ (B, n_heads, head_size, seq_len) -> (B, n_heads, seq_len, seq_len)

        # Scale
        attention = attention * self.scale

        # Apply attention mask
        if mask is not None:
            while len(mask.shape) < len(attention.shape):
                mask = mask[:, None, ...]

            attention = attention.masked_fill(mask == 0, float("-inf"))

        # Applying softmax
        attention = torch.softmax(attention, dim=-1)

        # Get dot product with values
        attention = torch.matmul(attention, V) # (B, n_heads, seq_len, seq_len) @ (B, n_heads, seq_len, head_size) -> (B, n_heads, seq_len, head_size)

        # Combine heads
        attention = attention.transpose(1, 2) # (B, n_heads, seq_len, head_size) -> (B, seq_len, n_heads, head_size)
        attention = attention.contiguous().view(x.shape) # (B, seq_len, n_heads, head_size) -> (B, seq_len, C)

        # Output projection
        attention = self.out_proj(attention)

        # Dropout
        attention = self.dropout(attention)

        return attention

class TransformerBlock(nn.Module):
    def __init__(self,
                 model_width,      # Width of model
                 cond_width=None,  # Width of condition input
                 n_heads=1,        # Number of attention heads
                 dropout=0.,       # Dropout rate
                 r_mlp=4,          # Ratio to calculate MLP hidden size
                 bias=False        # Linear layer bias
            ):
        super().__init__()

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(model_width)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(
            model_width,
            cond_width if cond_width is not None else model_width,
            n_heads,
            dropout=dropout,
            bias=bias
        )

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(model_width)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(model_width, model_width * r_mlp, bias=bias),
            nn.GELU(),
            nn.Linear(model_width * r_mlp, model_width, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, cond=None):
        # Residual Connection After Sub-Layer 1
        x = x + self.mha(self.ln1(x), mask=mask, cond=cond)

        # Residual Connection After Sub-Layer 2
        x = x + self.mlp(self.ln2(x))

        return x