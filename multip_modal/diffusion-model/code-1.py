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

imgs = []

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)
    
    beta = betas[t]
    eps = torch.randn_like(x) # 生成和x形状相同的噪声
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

# 2行5列的方式显示10张图片
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Noise: {i * 100}")
    plt.axis("off")

plt.show()
    