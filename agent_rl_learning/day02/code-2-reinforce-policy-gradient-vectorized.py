"""Train a CartPole policy with the Monte Carlo REINFORCE algorithm.

Difference from code-2-old.py:

* Both files are policy-gradient/REINFORCE implementations.
* code-2-old.py calculates one full-trajectory return G_0 and uses that same
  value to weight every log pi(A_t|S_t). This is a valid but higher-variance
  Monte Carlo policy-gradient estimator.
* This file calculates a separate reward-to-go G_t for every time step and
  pairs it with the matching log pi(A_t|S_t). An action is therefore weighted
  only by rewards received from that step onward, which generally reduces
  variance.
* This file also uses the current Gymnasium reset, step, and rendering APIs;
  code-2-old.py preserves the legacy Gym version for comparison.
"""

import argparse

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def save_animation(images: list[np.ndarray]) -> None:
    """Save the frames from one test episode as MP4 and GIF animations."""
    figure, axis = plt.subplots(figsize=(5, 3))
    animation_frames = []

    for step, image in enumerate(images, start=1):
        frame = [
            axis.imshow(image, animated=True),
            axis.text(10, 20, f"Step: {step}", color="black", animated=True),
        ]
        animation_frames.append(frame)

    axis.axis("off")
    result = animation.ArtistAnimation(
        figure,
        animation_frames,
        interval=100,
        blit=True,
    )
    result.save("new-cartpole.mp4", writer="ffmpeg")
    result.save("new-cartpole.gif", writer="pillow")
    plt.close(figure)


def plot_returns(episode_returns: list[float], filename: str) -> None:
    """Plot the total undiscounted reward obtained in each episode."""
    figure, axis = plt.subplots()
    episodes = np.arange(1, len(episode_returns) + 1)

    axis.plot(episodes, episode_returns, alpha=0.45, label="Episode return")

    # A moving average makes the overall learning trend easier to see.
    window_size = min(50, len(episode_returns))
    if window_size > 1:
        moving_average = np.convolve(
            episode_returns,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        axis.plot(
            episodes[window_size - 1 :],
            moving_average,
            label=f"{window_size}-episode average",
        )

    axis.set(xlabel="Episode", ylabel="Return", title="REINFORCE on CartPole-v1")
    axis.legend()
    figure.savefig(filename, bbox_inches="tight")
    plt.close(figure)


class PolicyNet(nn.Module):
    """Map a four-dimensional CartPole state to two action scores (logits)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        # Categorical can consume logits directly, so softmax is not needed here.
        return self.output(hidden)


class Agent:
    """A stochastic policy trained using Monte Carlo policy gradients."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98  # Discount factor: how strongly future rewards matter.
        self.policy = PolicyNet(state_size, action_size)
        # The loss is averaged over an episode, so a modest 2e-3 rate works well.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.002)

    def get_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Sample from pi(.|s), or choose its most likely action when testing."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state_tensor).squeeze(0)

        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        *,
        seed: int | None = None,
    ) -> tuple[list[np.ndarray], list[int], list[float]]:
        """Run one episode and collect (S_t, A_t, R_t) at every time step."""
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        return states, actions, rewards

    def discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        """Compute G_t = R_t + gamma*R_(t+1) + ... for every time step."""
        returns = []
        return_from_t = 0.0

        # Work backward because G_t = R_t + gamma * G_(t+1).
        for reward in reversed(rewards):
            return_from_t = reward + self.gamma * return_from_t
            returns.append(return_from_t)

        # The loop generated [G_(T-1), ..., G_0], so restore chronological order.
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def update(
        self,
        trajectory: tuple[list[np.ndarray], list[int], list[float]],
    ) -> float:
        """Perform one REINFORCE update using a complete sampled episode."""
        states, actions, rewards = trajectory
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        returns = self.discounted_returns(rewards)

        # log pi_theta(A_t|S_t) for each action that was actually taken.
        distribution = Categorical(logits=self.policy(state_tensor))
        # Correct—I replaced gather() with PyTorch’s Categorical.log_prob():
        selected_log_probabilities = distribution.log_prob(action_tensor)

        # Each action is weighted by its OWN future return G_t, not just G_0.
        # Dividing by episode length keeps the gradient scale reasonably stable.
        loss = -(selected_log_probabilities * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def train_agent(agent: Agent, env: gym.Env, episodes: int, seed: int) -> list[float]:
    """Train the agent and return the total reward from each episode."""
    episode_returns = []

    for episode in range(episodes):
        # Seeding the first reset makes an experiment reproducible while allowing
        # Gymnasium's random-number generator to advance in later episodes.
        episode_seed = seed if episode == 0 else None
        trajectory = agent.collect_trajectory(env, seed=episode_seed)
        loss = agent.update(trajectory)
        episode_return = sum(trajectory[2])
        episode_returns.append(episode_return)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"回合：{episode + 1:4d}，总奖励：{episode_return:5.1f}，"
                f"损失：{loss:8.4f}"
            )

    recent_count = min(100, len(episode_returns))
    recent_average = float(np.mean(episode_returns[-recent_count:]))
    print(f"最近 {recent_count} 回合的平均奖励：{recent_average:.1f}")
    return episode_returns


def test_agent(agent: Agent, seed: int) -> list[np.ndarray]:
    """Run the trained policy once and return RGB frames for visualization."""
    # Gymnasium requires render_mode when the environment is created.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames: list[np.ndarray] = []

    try:
        state, _ = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action = agent.get_action(state, greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip creation of cartpole.mp4 and cartpole.gif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    training_env = gym.make("CartPole-v1")
    agent = Agent(
        state_size=training_env.observation_space.shape[0],
        action_size=training_env.action_space.n,
    )

    try:
        episode_returns = train_agent(agent, training_env, args.episodes, args.seed)
    finally:
        training_env.close()

    plot_returns(episode_returns, "pg-loss-CartPole-v1.pdf")

    if not args.no_animation:
        save_animation(test_agent(agent, args.seed + 1))


if __name__ == "__main__":
    main()
