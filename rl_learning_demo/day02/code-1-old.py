import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # the state is 4 dim
        self.l1 = nn.Linear(4, 128)
        # action_size is two: go to left or right
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2
        # 初始化策略网络π_θ
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optim.Adam(
            self.pi.parameters(),
            lr=self.lr
        )

    def get_action(self, state):
        """输入参数：环境的状态"""
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()

        return action, probs


env = gym.make("CartPole-v0")
state = env.reset()
agent = Agent()

action, probs = agent.get_action(state)
print("采取的动作：", "向左推" if action == 0 else "向右推")
print("采取的动作的概率：", probs[action].item())

G = 1.0  # G(τ)
J = -G * probs[action].log()  # −𝐺(𝜏)log𝜋𝜃(𝐴0|𝑆0)
J.backward()
agent.optimizer.step()

_, probs = agent.get_action(state)
print("采取的动作：", "向左推" if action == 0 else "向右推")
print("采取的动作的概率：", probs[action].item())
