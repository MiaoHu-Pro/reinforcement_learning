import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image = plt.imread("./flower.png")
print(image.shape)

preprocess = transforms.ToTensor()
x = preprocess(image)
print(image.shape)


def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


# 最大时间步
T = 1000
# 方差计划的起始值
beta_start = 0.0001
# 方差计划的结束值
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
print(betas)

# 一步得到x_t,使用闭式解（closed form）


def add_noise(x_0, t, betas):
    T = len(betas)

    alphas = 1 - betas  # [α_1, α_2, ...]
    # cumprod功能：[1,2,3,4] --> [1,2,6,24]
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]  # alpha_bar_t

    eps = torch.randn_like(x_0)
    # 闭式解
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar)*eps
    
    return x_t

t = 10
x_t = add_noise(x, t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.show()