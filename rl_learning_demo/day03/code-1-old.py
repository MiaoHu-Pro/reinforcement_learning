# actor-critic的实现

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


def show_animation(imgs):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    text = ax.text(10, 20, "", fontsize=12, color="black")

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # Step数表示
        frames.append(frame)

    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")
    ani.save("cartpole.gif", writer="pillow")

    plt.close(fig)
    return ani


def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


class PolicyNet(nn.Module):
    """策略神经网络的结构"""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):  # x是S_t
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    """价值函数神经网络V_ω"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.005
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs

    def update(self,
               state, # 𝑆_𝑡
               next_state, # 𝑆_𝑡+1
               reward, # 𝑅𝑡
               action_prob, # 𝜋𝜃(𝐴_𝑡|𝑆_𝑡)
               done):

        state = torch.tensor(state).unsqueeze(0) # 𝑆_𝑡
        next_state = torch.tensor(next_state).unsqueeze(0) # 𝑆_𝑡+1

        # ① 计算价值网络的损失  self.v的损失：均方差
        # td误差 δ = R_t + γV(S_(t+1)) - V(S_t)
        # TD target = δ + V(S_t)
        # 而对于价值函数𝑉𝜔，则通过TD方法，以接近𝑅_𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1)为目标训练𝑉𝜔(𝑆_𝑡)这个神经网络。
        # TD目标 = R_t + γV(S_(t+1)) -->  （G_t+1）

        #
        # Definitions:
        #
        #   $$
        #   y_t = R_t + \gamma (1-d_t)V(S_{t+1})
        #   $$
        #
        #   Here, (y_t) is the TD target.
        #
        #   The TD error is:
        #
        #   $$
        #   \delta_t = y_t - V(S_t)
        #   $$
        #
        #   Therefore:
        #
        #   $$
        #   y_t = \delta_t + V(S_t)
        #   $$
        #
        #   So the comment TD target = δ + V(S_t) is mathematically correct.

        # One-step TD target: 𝑅𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1)
        # y_t = R_t + gamma * (1 - done) * V(S_{t+1})
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target = target.detach()  # 从计算图剥离，变成一个常数 target = target.detach()
        # Current value estimate: V(S_t)
        value = self.v(state) # 𝑉𝜔(𝑆_𝑡)
        loss_fn = nn.MSELoss()  # (𝑅𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1) − 𝑉𝜔(𝑆_𝑡))2 → 0
        # Train the critic so that V(S_t) approaches the TD target y_t
        loss_v = loss_fn(value, target)

        # ② 计算策略网络的损失
        # TD error:
        # td_error: delta_t = y_t - V(S_t)
        delta = target - value  # # 𝛿 = 𝑅𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1) − 𝑉𝜔(𝑆𝑡) 必须从计算图中剥离出来
        # Actor-Critic loss:  −(𝑅_𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1) − 𝑉𝜔(𝑆_𝑡)) log 𝜋𝜃(𝐴_𝑡|𝑆_𝑡)
        loss_pi = -torch.log(action_prob) * delta.detach().item()  #

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_v.backward()  # ∇𝜔(𝑅_𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1) − 𝑉𝜔(𝑆_𝑡))2
        loss_pi.backward() # −(𝑅_𝑡 + 𝛾𝑉𝜔(𝑆_𝑡+1) − 𝑉𝜔(𝑆_𝑡))∇𝜃 log 𝜋𝜃(𝐴_𝑡|𝑆_𝑡)
        self.optimizer_pi.step()
        self.optimizer_v.step()


env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    state = env.reset()  # S_0
    done = False
    total_reward = 0.0

    while not done:
        action, probs = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        # 执行一个动作，更新一次策略网络和价值网络
        agent.update(state, next_state, reward, probs[action], done)

        state = next_state
        total_reward += reward

    return_list.append(total_reward)
    episode_list.append(episode)
    if episode % 100 == 0:
        print(f"回合：{episode}, 总奖励：{total_reward}")

plot_loss(episode_list, return_list, "actor-critic-pg-loss.pdf")
