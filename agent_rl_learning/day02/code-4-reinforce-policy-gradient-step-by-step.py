"""Train CartPole with step-by-step reward-to-go REINFORCE.

This is the Gymnasium/PyTorch rewrite of ``code-4-old.py``. The old file is
preserved for comparison and must be run in ``old_version_rf_env``.

The central REINFORCE update is deliberately written as a backward Python loop:

    G_t = R_t + gamma * G_(t+1)
    loss += -G_t * log pi_theta(A_t | S_t)

The policy weights stay fixed while one complete trajectory is collected. Once
the episode ends, one loss is built, ``backward()`` computes its gradients, and
``optimizer.step()`` updates the weights once.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


Trajectory = tuple[list[np.ndarray], list[int], list[float]]


class PolicyNet(nn.Module):
    """Map a CartPole state to unnormalized scores for left and right."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        # Return logits. Categorical performs the stable softmax internally.
        return self.output(hidden)


class ReinforceAgent:
    """A stochastic policy trained from one complete trajectory at a time."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.policy = PolicyNet(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.002)

    def choose_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Sample an action during training or choose the best action in testing."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        # Trajectory collection does not need a computation graph. The states and
        # selected actions are saved, then evaluated again inside update().
        with torch.no_grad():
            logits = self.policy(state_tensor)

        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        *,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect one trajectory from step 0 to the final step of one episode."""
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store the transition information needed by the policy update.
            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        return states, actions, rewards

    def update(self, trajectory: Trajectory) -> float:
        """Build one REINFORCE loss and update the policy once."""
        states, actions, rewards = trajectory
        loss_terms: list[torch.Tensor] = []
        return_from_t = 0.0

        # Traverse the complete trajectory backward. This computes a different
        # reward-to-go G_t for every action instead of applying only G_0 to all
        # actions. Each action is influenced only by rewards from its step onward.
        for reward, state, action in zip(
            reversed(rewards),
            reversed(states),
            reversed(actions),
        ):
            return_from_t = reward + self.gamma * return_from_t

            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.int64)

            # This forward pass tracks gradients. Unlike code-4-old.py, it does
            # not sample a second, unused action while reconstructing the loss.
            distribution = Categorical(logits=self.policy(state_tensor))
            selected_log_probability = distribution.log_prob(action_tensor)
            loss_terms.append(-return_from_t * selected_log_probability)

        # Averaging makes the gradient scale less sensitive to episode length.
        loss = torch.stack(loss_terms).mean()

        self.optimizer.zero_grad()  # Remove gradients from the previous episode.
        loss.backward()             # Calculate d(loss)/d(policy weights).
        self.optimizer.step()       # Update the policy weights exactly once.
        return float(loss.item())


def train(
    agent: ReinforceAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> list[float]:
    """Collect one trajectory and perform one update in every episode."""
    episode_returns: list[float] = []

    for episode in range(episodes):
        # Seed only the first reset; later resets advance the environment RNG.
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


def plot_returns(episode_returns: list[float], output_path: Path) -> None:
    """Save episode returns and a moving-average learning curve."""
    figure, axis = plt.subplots()
    episode_numbers = np.arange(1, len(episode_returns) + 1)
    axis.plot(episode_numbers, episode_returns, alpha=0.4, label="Episode return")

    window_size = min(50, len(episode_returns))
    if window_size > 1:
        moving_average = np.convolve(
            episode_returns,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        axis.plot(
            episode_numbers[window_size - 1 :],
            moving_average,
            label=f"{window_size}-episode average",
        )

    axis.set(
        xlabel="Episode",
        ylabel="Return",
        title="Reward-to-go REINFORCE on CartPole-v1",
    )
    axis.legend()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def evaluate(agent: ReinforceAgent, seed: int) -> list[np.ndarray]:
    """Run one greedy test episode and capture its RGB frames."""
    # Gymnasium requires render_mode when constructing the environment.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames: list[np.ndarray] = []

    try:
        state, _ = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action = agent.choose_action(state, greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    return frames


def save_animation(frames: list[np.ndarray], output_dir: Path) -> None:
    """Save a test trajectory as GIF and MP4 files."""
    figure, axis = plt.subplots(figsize=(5, 3))
    artists = []

    for step, frame in enumerate(frames, start=1):
        artists.append(
            [
                axis.imshow(frame, animated=True),
                axis.text(10, 20, f"Step: {step}", animated=True),
            ]
        )

    axis.axis("off")
    result = animation.ArtistAnimation(figure, artists, interval=100, blit=True)
    result.save(output_dir / "code-4-cartpole.mp4", writer="ffmpeg")
    result.save(output_dir / "code-4-cartpole.gif", writer="pillow")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip the final rendered GIF and MP4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for the learning curve and animations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    training_env = gym.make("CartPole-v1")
    training_env.action_space.seed(args.seed)
    agent = ReinforceAgent(
        state_size=training_env.observation_space.shape[0],
        action_size=training_env.action_space.n,
    )

    try:
        episode_returns = train(agent, training_env, args.episodes, args.seed)
    finally:
        training_env.close()

    plot_returns(
        episode_returns,
        args.output_dir / "code-4-reinforce-returns.pdf",
    )

    if not args.no_animation:
        frames = evaluate(agent, seed=args.seed + 1)
        save_animation(frames, args.output_dir)


if __name__ == "__main__":
    main()
