import torch
import numpy as np


def tokenizer(text, mask=None, text_seq_length=128):
    """UTF-8 byte tokenizer with fixed length and explicit BOS/EOS bytes."""
    # If a mask is not inputted it encodes text, otherwise it decodes the text
    if mask is None:
        if text_seq_length < 2:
            raise ValueError("text_seq_length must leave room for BOS and EOS")
        # Token IDs are UTF-8 bytes. Truncate bytes (not Python characters),
        # then always retain EOS and pad to one stable tensor length.
        content = str(text).encode("utf-8")[: text_seq_length - 2]
        encoded = bytes([2]) + content + bytes([3])
        encoded += bytes(text_seq_length - len(encoded))
        out = torch.tensor(list(encoded), dtype=torch.long)

        # Create text mask
        mask = out.ne(0).to(torch.long)
    else:
        # Decode text
        valid_length = int(mask.sum().item())
        content = bytes(int(x) for x in text[1:max(1, valid_length - 1)])
        out = content.decode("utf-8", errors="replace")
        mask = None

    return out, mask

# Gets mean and standard deviation of a set of images


def get_mean_std(data, img_channels, denom=1):
    # Get only the images from the dataset
    try:
        images = np.array([x["image"] for x in data]) / denom
    except:
        images = np.array([x[0] for x in data]) / denom

    # Combine pixels of each channel into one dimension
    images = images.reshape(img_channels, -1)

    # Calculate the mean and standard deviation
    mean, std = images.mean(axis=1), images.std(axis=1)

    return mean, std

# Returns beta schedule


def get_beta_schedule(schedule="linear", max_time=1000, s=0.008):
    """生成方差计划β_t的列表"""
    if schedule == "linear":
        scale = 1000 / max_time
        betas = torch.linspace(1e-4 * scale, 0.02 * scale, max_time)
    elif schedule == "cosine":
        t = torch.linspace(0, max_time, max_time + 1)
        a_bars = torch.cos((((t / max_time) + s) / (1 + s)) * (np.pi / 2)) ** 2
        a_bars = a_bars / a_bars[0]
        betas = 1 - (a_bars[1:] / a_bars[:-1])
        betas = torch.clamp(betas, min=0, max=0.999)
    else:
        raise ValueError(f"Beta schedule not implemented: {schedule}")

    return betas


def get_schedule_values(schedule="linear", max_time=1000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """计算添加噪声和去除噪声用到的所有的计划值"""
    schedule_values = {}
    schedule_values["betas"] = get_beta_schedule(
        schedule, max_time).to(device)  # β_t组成的列表
    schedule_values["alphas"] = 1.0 - schedule_values["betas"]  # α_t组成的列表
    schedule_values["alpha_bars"] = torch.cumprod(
        schedule_values["alphas"], axis=0)  # α_bar_t组成的算子
    schedule_values["sqrt_recip_alphas"] = torch.sqrt(
        1.0 / schedule_values["alphas"])  # 1/sqrt(α_t)
    schedule_values["sqrt_alpha_bars"] = torch.sqrt(
        schedule_values["alpha_bars"])  # sqrt(α_bar_t)
    schedule_values["sqrt_one_minus_alpha_bars"] = torch.sqrt(
        1.0 - schedule_values["alpha_bars"])
    schedule_values["alpha_bars_prev"] = torch.cat(
        (torch.ones(1, device=device), schedule_values["alpha_bars"][:-1]))
    schedule_values["posterior_variance"] = schedule_values["betas"] * \
        (1.0 - schedule_values["alpha_bars_prev"]) / \
        (1.0 - schedule_values["alpha_bars"])
    # Reverse diffusion multiplies standard-normal noise by standard deviation,
    # not variance. The original implementation used variance directly.
    schedule_values["sigma"] = torch.sqrt(
        torch.clamp(schedule_values["posterior_variance"], min=0.0)
    )
    return schedule_values


def extract_and_expand(x, idx, shape):
    return x[idx].reshape(idx.shape[0], *((1, ) * (len(shape) - 1)))


def freeze_model(model, set_eval=True):
    if set_eval:
        model.eval()

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def forward_diffusion(x_0, schedule_values, t):
    """前向扩散，使用解析解一步得到x_t图片"""
    """用在两个地方，1. 训练先验模型时给clip图片特征向量添加噪声，2. 训练条件扩散模型decoder时，给图片添加噪声"""
    noise = torch.randn_like(x_0)
    sqrt_alpha_bars = extract_and_expand(
        schedule_values["sqrt_alpha_bars"], t, x_0.shape)
    sqrt_one_minus_alpha_bars = extract_and_expand(
        schedule_values["sqrt_one_minus_alpha_bars"], t, x_0.shape)
    x_noisy = (sqrt_alpha_bars * x_0) + (sqrt_one_minus_alpha_bars * noise)
    return x_noisy, noise
