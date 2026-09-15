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
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")

# 策略网络（演员）


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self):
        self.lr = 0.0002
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()

        return action, probs

    def collect_trajectory(self, env):
        """采样一条轨迹"""
        state = env.reset()
        states, log_probs, actions = [], [], []
        episode_reward = 0
        done = False

        while not done:
            action, probs = self.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_prob = torch.log(probs)[action]
            log_probs.append(log_prob.item())

            state = next_state
            episode_reward += reward

        # 归一化奖励
        normalized_reward = episode_reward / 200.0

        return states, log_probs, actions, normalized_reward

    def update(self, trajectories):
        advantages = calc_advantages_with_grpo(trajectories)

        for step in range(20):
            loss = 0.0
            for traj, advantage in zip(trajectories, advantages):
                """遍历组里面的每一条轨迹和对应的组内优势"""
                states, log_probs, actions, _ = traj
                states = torch.tensor(states)
                # 旧策略，每个动作的log prob
                log_probs = torch.tensor(log_probs).view(-1, 1)
                actions = torch.tensor(actions).view(-1, 1)
                # 相同状态在新策略下的，每个动作的log prob
                new_log_probs = torch.log(self.pi(states).gather(1, actions))
                # 计算比值，使用e^(logA - logB)
                ratio = torch.exp(new_log_probs - log_probs)
                # 裁剪
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                traj_loss = torch.mean(
                    -torch.min(ratio * advantage, clipped_ratio * advantage))

                loss += traj_loss
            loss = loss / len(trajectories)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return None


def calc_advantages_with_grpo(trajectories):
    """使用一组轨迹计算某条轨迹的组内优势"""
    # [轨迹0的归一化奖励，轨迹1的归一化奖励，...]
    rewards = [r for o, l, a, r in trajectories]
    mean_reward = sum(rewards) / len(rewards)
    std_reward = np.std(rewards) + 1e-8
    # [轨迹0的组相对优势，轨迹1的组相对优势，...]
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    return advantages


def train(agent, env):
    G = 5  # 一组轨迹有5条
    trial_num = 0
    while True:
        for episode in range(20):
            trajectories, episode_rewards = [], []
            for _ in range(G):
                states, log_probs, actions, normalized_reward = agent.collect_trajectory(
                    env)
                trajectories.append(
                    (states, log_probs, actions, normalized_reward))
                episode_rewards.append(normalized_reward * 200)
            agent.update(trajectories)

        # 一组轨迹的平均奖励
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        trial_num += 1

        if avg_reward > 195:
            print("训练结束，训练回合数：", trial_num)
            return
        else:
            print(f"训练回合数：{trial_num}，平均奖励：{avg_reward}")


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


def main():
    env = gym.make("CartPole-v0")
    agent = Agent()
    train(agent, env)
    test_agent(agent, env)


if __name__ == "__main__":
    main()
