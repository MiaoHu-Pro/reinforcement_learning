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


class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子γ
        self.lr = 0.0002  # 学习率α
        self.action_size = 2  # 两个动作：向左推和向右推
        # 初始化策略网络π_θ
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optim.Adam(
            self.pi.parameters(),
            lr=self.lr
        )

    def get_action(self, state):
        """输入参数：环境的状态"""
        # 动作的概率分布
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # 创建一个二项分布采样器
        m = Categorical(probs)
        # 采样一个动作
        action = m.sample().item()

        return action, probs

    def collect_trajectory(self, env):
        """在环境env中采样一条轨迹"""
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            # 采取动作A_t
            action, _ = self.get_action(state)
            # 执行动作A_t
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # 状态转移
            state = next_state
        # states: [S_0, S_1, S_2, ..., S_T]
        # actions: [A_0, A_1, A_2, ..., A_T]
        # rewards: [R_0, R_1, R_2, ..., R_T]
        return states, actions, rewards

    def update(self, trajectory):
        """用轨迹trajectory数据更新策略网络"""
        states, actions, rewards = trajectory
        # # 逆序计算G(τ)
        # G = 0
        # for r in rewards[::-1]:
        #     G = r + self.gamma * G
        # # 计算损失
        # loss = 0
        # for s, a in zip(states, actions):
        #     # S_t --> 动作的概率分布
        #     _, probs = self.get_action(s)
        #     # 计算动作的对数概率
        #     # 使用轨迹数据中S_t对应的A_t，获取A_t的概率
        #     log_prob = torch.log(probs)[a]
        #     # 计算损失
        #     loss += -log_prob * G
        G, loss = 0, 0
        for r, s, a in zip(rewards[::-1], states[::-1], actions[::-1]):
            # 计算G_t
            G = r + self.gamma * G
            _, probs = self.get_action(s)
            log_prob = torch.log(probs)[a]
            loss += - G * log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    # 采样一条轨迹
    trajectory = agent.collect_trajectory(env)
    # 使用采样到的轨迹更新策略
    agent.update(trajectory)

    # 统计数据
    rewards = trajectory[2]
    return_list.append(sum(rewards))
    episode_list.append(episode)

    if episode % 100 == 0:
        print("回合：{}, 总奖励：{:.1f}".format(episode, sum(rewards)))

# 可视化损失
plot_loss(episode_list, return_list, "reinforce-pg-loss.pdf")


def test_agent(agent, env):
    """测试训练后的智能体"""
    state = env.reset()
    done = False
    frames = []

    while not done:
        frames.append(env.render(mode="rgb_array"))
        action, _ = agent.get_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

    env.close()
    show_animation(frames)


test_agent(agent, env)
