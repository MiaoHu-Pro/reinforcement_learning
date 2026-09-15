import torch
import torch.nn as nn

x = torch.tensor([1.0])            # 形状 [N, 1]，每行一个标量输入
# w = torch.randn(1, 1, requires_grad=True)  # 标量权重（等价于单个参数）
w = torch.tensor([2.0], requires_grad=True)
# b = torch.randn(1, requires_grad=True)     # 标量偏置
b = torch.tensor([3.0], requires_grad=True)

# 前向：y = sigmoid(w*x + b)
y = torch.sigmoid(x @ w + b)     # 结果形状 [N, 1]
# 构造标量损失以便反向传播
target = torch.tensor([2.0])
loss = nn.MSELoss()(y, target)

# 反向传播
loss.backward()

print("loss:", loss.item())
print("w:", w.item(), "b:", b.item())
print("w.grad:", w.grad.item(), "b.grad:", b.grad.item())
