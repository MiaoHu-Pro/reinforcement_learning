"""Train CartPole with Actor-Critic and an explicit one-step TD error.

For every transition, the critic computes

    delta_t = R_t + gamma * V_phi(S_(t+1)) - V_phi(S_t)

The critic trains directly from this one-step error. The actor combines the
same TD errors with generalized advantage estimation (GAE), which propagates
credit across future steps while remaining TD-error-based. Transitions are
collected for one episode and updated as one batch.
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


Trajectory = tuple[
    list[np.ndarray],  # S_t
    list[int],         # A_t
    list[float],       # R_t
    list[np.ndarray],  # S_(t+1)
    list[bool],        # true task termination flags
]


class ActorNet(nn.Module):
    """Represent the stochastic policy pi_theta(a|s)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden)


class CriticNet(nn.Module):
    """Approximate V_phi(s) within CartPole's valid discounted-value range."""

    def __init__(self, state_size: int, maximum_value: float):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)
        self.maximum_value = maximum_value

        # Start with a small positive value estimate rather than the midpoint of
        # [0, maximum_value]. The critic can then grow as episodes improve.
        nn.init.constant_(self.output.bias, -3.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        raw_value = self.output(hidden).squeeze(-1)
        return self.maximum_value * torch.sigmoid(raw_value)


class OneStepTDAgent:
    """Train an actor and critic from one-step temporal-difference errors."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gae_lambda: float,
    ):
        self.gamma = 0.98
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 0.01
        self.actor = ActorNet(state_size, action_size)
        # CartPole gives at most 1 reward per step, so its infinite-horizon
        # discounted return is bounded by 1 / (1 - gamma) = 50.
        maximum_value = 1.0 / (1.0 - self.gamma)
        self.critic = CriticNet(state_size, maximum_value)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def choose_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Sample during training or choose the most likely test action."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state_tensor)

        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        *,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect all one-step transitions from one on-policy episode."""
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[np.ndarray] = []
        terminations: list[bool] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            next_states.append(next_state)
            terminations.append(bool(terminated))
            state = next_state

        return states, actions, rewards, next_states, terminations

    def update(self, trajectory: Trajectory) -> tuple[float, float, float]:
        """Update both networks using explicit one-step TD errors."""
        states, actions, rewards, next_states, terminations = trajectory
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
        )
        terminated_tensor = torch.tensor(terminations, dtype=torch.float32)

        distribution = Categorical(logits=self.actor(state_tensor))
        selected_log_probabilities = distribution.log_prob(action_tensor)
        entropy = distribution.entropy().mean()
        values = self.critic(state_tensor)

        # The target is fixed during this update. There must be no gradient
        # through V_phi(S_(t+1)); this is the semi-gradient TD method.
        with torch.no_grad():
            next_values = self.critic(next_state_tensor)

            # Bootstrap on ordinary and time-limit-truncated transitions.
            # Only a true task termination has zero future value.
            bootstrap_mask = 1.0 - terminated_tensor
            td_targets = (
                reward_tensor
                + self.gamma * bootstrap_mask * next_values
            )

        # This is exactly:
        # delta_t = R_t + gamma*V_phi(S_(t+1)) - V_phi(S_t)
        td_errors = td_targets - values

        # Build GAE from the explicit one-step TD errors:
        # A_t = delta_t + gamma*lambda*A_(t+1).
        # This carries credit farther than one transition without changing the
        # critic's one-step TD target.
        gae_advantages = []
        advantage_from_t = 0.0
        for td_error in reversed(td_errors.detach()):
            advantage_from_t = (
                td_error + self.gamma * self.gae_lambda * advantage_from_t
            )
            gae_advantages.append(advantage_from_t)
        gae_advantages.reverse()
        actor_advantages = torch.stack(gae_advantages)

        # Standardization reduces the variance and scale sensitivity of the
        # actor update. The critic still receives original one-step TD targets.
        if actor_advantages.numel() > 1:
            actor_advantages = (
                actor_advantages - actor_advantages.mean()
            ) / (actor_advantages.std(unbiased=False) + 1e-8)

        # The actor increases the probability of actions with positive
        # standardized GAE advantage and decreases it for negative advantage.
        # detach() above prevents actor_loss from training the critic.
        actor_loss = -(
            selected_log_probabilities * actor_advantages
        ).mean() - self.entropy_coefficient * entropy

        # Huber regression is robust to occasional large bootstrapped errors.
        critic_loss = F.smooth_l1_loss(values, td_targets)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(td_errors.detach().mean().item()),
        )


def train(
    agent: OneStepTDAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Collect one episode, then batch-update all of its one-step TD errors."""
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(episodes):
        episode_seed = seed if episode == 0 else None
        trajectory = agent.collect_trajectory(env, seed=episode_seed)
        actor_loss, critic_loss, mean_td_error = agent.update(trajectory)
        episode_return = sum(trajectory[2])

        episode_returns.append(episode_return)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"回合：{episode + 1:4d}，总奖励：{episode_return:5.1f}，"
                f"策略损失：{actor_loss:8.4f}，价值损失：{critic_loss:8.4f}，"
                f"平均TD误差：{mean_td_error:7.3f}"
            )

    recent_count = min(100, len(episode_returns))
    recent_average = float(np.mean(episode_returns[-recent_count:]))
    print(f"最近 {recent_count} 回合的平均奖励：{recent_average:.1f}")
    return episode_returns, actor_losses, critic_losses


def plot_training(
    episode_returns: list[float],
    actor_losses: list[float],
    critic_losses: list[float],
    output_path: Path,
) -> None:
    """Save return, actor-loss, and critic-loss curves."""
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
    axes[0].set_title("Actor-Critic with one-step TD errors")
    axes[1].plot(episode_numbers, actor_losses)
    axes[1].set_ylabel("Actor loss")
    axes[2].plot(episode_numbers, critic_losses)
    axes[2].set(xlabel="Episode", ylabel="Critic loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def evaluate(agent: OneStepTDAgent, seed: int) -> list[np.ndarray]:
    """Run one greedy evaluation episode and collect RGB frames."""
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
    """Save a greedy evaluation episode as GIF and MP4 animations."""
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
    result.save(output_dir / "code-7-one-step-td-cartpole.mp4", writer="ffmpeg")
    result.save(output_dir / "code-7-one-step-td-cartpole.gif", writer="pillow")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help=(
            "TD-error trace parameter; use 0 for the direct one-step delta, "
            "or 0.95 for more stable multi-step credit assignment"
        ),
    )
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
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    training_env = gym.make("CartPole-v1")
    training_env.action_space.seed(args.seed)
    agent = OneStepTDAgent(
        state_size=training_env.observation_space.shape[0],
        action_size=training_env.action_space.n,
        gae_lambda=args.gae_lambda,
    )

    try:
        episode_returns, actor_losses, critic_losses = train(
            agent,
            training_env,
            args.episodes,
            args.seed,
        )
    finally:
        training_env.close()

    plot_training(
        episode_returns,
        actor_losses,
        critic_losses,
        args.output_dir / "code-7-one-step-td-training.pdf",
    )

    if not args.no_animation:
        frames = evaluate(agent, seed=args.seed + 1)
        save_animation(frames, args.output_dir)


if __name__ == "__main__":
    main()
