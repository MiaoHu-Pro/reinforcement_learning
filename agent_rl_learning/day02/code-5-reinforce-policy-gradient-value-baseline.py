"""Train CartPole with REINFORCE and a learned state-value baseline.

This is the Gymnasium/PyTorch rewrite of ``code-5-old.py``:

* The old file uses a fixed baseline: ``advantage = G_t - 5.0``.
* This file learns a neural network ``V_phi(S_t)`` and uses
  ``advantage = G_t - V_phi(S_t)``.
* The policy and value networks have separate losses and optimizers.

The learned value estimate can be imperfect (and therefore biased as an
estimate of the true value), but an action-independent baseline does not bias
the expected policy gradient when it is detached from the policy loss. Its
main benefit is reducing the variance of policy-gradient updates.
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
    """Actor: map a state to logits for the two CartPole actions."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden)


class ValueNet(nn.Module):
    """Critic/baseline: estimate the scalar state value V_phi(S_t)."""

    def __init__(self, state_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        # squeeze(-1) changes [T, 1] into [T], matching the returns tensor.
        return self.output(hidden).squeeze(-1)


class ReinforceWithBaselineAgent:
    """REINFORCE agent with a neural-network state-value baseline."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.policy = PolicyNet(state_size, action_size)
        self.value = ValueNet(state_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.002)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=0.002)

    def choose_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Sample from the policy during training or act greedily in testing."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        # Do not retain one autograd graph for the whole environment interaction.
        # update() will evaluate all stored states again with gradients enabled.
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
        """Collect one complete trajectory from one CartPole episode."""
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        return states, actions, rewards

    def discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        """Calculate the Monte Carlo reward-to-go G_t for every time step."""
        returns = []
        return_from_t = 0.0

        # G_t = R_t + gamma * G_(t+1)
        for reward in reversed(rewards):
            return_from_t = reward + self.gamma * return_from_t
            returns.append(return_from_t)

        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, trajectory: Trajectory) -> tuple[float, float, float]:
        """Update the policy once and the value baseline once."""
        states, actions, rewards = trajectory
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)

        returns = self.discounted_returns(rewards)

        # ----- Forward passes with gradient tracking enabled -----
        distribution = Categorical(logits=self.policy(state_tensor))
        selected_log_probabilities = distribution.log_prob(action_tensor)

        value_estimates = self.value(state_tensor)

        # The advantage says whether the observed return was better or worse than
        # the value network expected from that state.
        # Adv_t = G_t - V_\phi(S_t)
        # The learned network `V_phi(s)` can be a biased or inaccurate approximation of
        # the true value function.
        advantages = returns - value_estimates

        # detach() is essential: policy_loss updates only the policy network.
        # The policy must treat the learned baseline as a fixed control variate.
        policy_loss = -(
            selected_log_probabilities * advantages.detach()
        ).mean()

        # Train V_phi(S_t) by regression toward the sampled Monte Carlo return G_t.
        value_loss = F.mse_loss(value_estimates, returns)

        # Clear gradients left by the previous episode.
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # The two graphs end at different networks because advantages are detached
        # in policy_loss. backward() computes gradients; it does not update weights.
        policy_loss.backward()
        value_loss.backward()

        # These are the two operations that actually change network parameters.
        self.policy_optimizer.step()
        self.value_optimizer.step()

        return (
            float(policy_loss.item()),
            float(value_loss.item()),
            float(advantages.detach().mean().item()),
        )


def train(
    agent: ReinforceWithBaselineAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Collect one trajectory and update both networks in every episode."""
    episode_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    for episode in range(episodes):
        episode_seed = seed if episode == 0 else None
        trajectory = agent.collect_trajectory(env, seed=episode_seed)
        policy_loss, value_loss, mean_advantage = agent.update(trajectory)

        episode_return = sum(trajectory[2])
        episode_returns.append(episode_return)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"回合：{episode + 1:4d}，总奖励：{episode_return:5.1f}，"
                f"策略损失：{policy_loss:8.4f}，价值损失：{value_loss:9.4f}，"
                f"平均优势：{mean_advantage:7.3f}"
            )

    recent_count = min(100, len(episode_returns))
    recent_average = float(np.mean(episode_returns[-recent_count:]))
    print(f"最近 {recent_count} 回合的平均奖励：{recent_average:.1f}")
    return episode_returns, policy_losses, value_losses


def plot_training(
    episode_returns: list[float],
    policy_losses: list[float],
    value_losses: list[float],
    output_path: Path,
) -> None:
    """Save the return curve plus actor and value-network losses."""
    episode_numbers = np.arange(1, len(episode_returns) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(episode_numbers, episode_returns, alpha=0.4)
    window_size = min(50, len(episode_returns))
    if window_size > 1:
        moving_average = np.convolve(
            episode_returns,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        axes[0].plot(
            episode_numbers[window_size - 1 :],
            moving_average,
            label=f"{window_size}-episode average",
        )
        axes[0].legend()

    axes[0].set_ylabel("Return")
    axes[0].set_title("REINFORCE with learned value baseline")
    axes[1].plot(episode_numbers, policy_losses)
    axes[1].set_ylabel("Policy loss")
    axes[2].plot(episode_numbers, value_losses)
    axes[2].set(xlabel="Episode", ylabel="Value MSE")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    agent: ReinforceWithBaselineAgent,
    seed: int,
) -> list[np.ndarray]:
    """Run one greedy test episode and capture RGB frames."""
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
    """Save the greedy test trajectory as GIF and MP4 animations."""
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
    result.save(output_dir / "code-5-cartpole.mp4", writer="ffmpeg")
    result.save(output_dir / "code-5-cartpole.gif", writer="pillow")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip creation of the final GIF and MP4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for the training plot and animations",
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
    agent = ReinforceWithBaselineAgent(
        state_size=training_env.observation_space.shape[0],
        action_size=training_env.action_space.n,
    )

    try:
        episode_returns, policy_losses, value_losses = train(
            agent,
            training_env,
            args.episodes,
            args.seed,
        )
    finally:
        training_env.close()

    plot_training(
        episode_returns,
        policy_losses,
        value_losses,
        args.output_dir / "code-5-baseline-reinforce-training.pdf",
    )

    if not args.no_animation:
        frames = evaluate(agent, seed=args.seed + 1)
        save_animation(frames, args.output_dir)


if __name__ == "__main__":
    main()
