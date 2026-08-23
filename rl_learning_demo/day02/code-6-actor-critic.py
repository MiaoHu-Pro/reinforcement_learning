"""Train CartPole with an n-step Actor-Critic algorithm.

The actor learns the stochastic policy pi_theta(a|s). The critic learns the
state-value function V_phi(s). Training collects a short rollout and uses an
n-step bootstrapped return to update both networks before the episode ends.

This is different from Monte Carlo REINFORCE, which waits for the complete
episode and calculates returns using only observed rewards.
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


class ActorNet(nn.Module):
    """Actor: represent pi_theta(a|s) with action logits."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden)


class CriticNet(nn.Module):
    """Critic: approximate the scalar state-value function V_phi(s)."""

    def __init__(self, state_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden).squeeze(-1)


class ActorCriticAgent:
    """Train separate actor and critic networks from short n-step rollouts."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.entropy_coefficient = 0.001
        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def choose_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Sample during training or choose the most likely action in testing."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        # The rollout itself does not retain a graph. update() evaluates the
        # stored states again and constructs one compact batched graph.
        with torch.no_grad():
            logits = self.actor(state_tensor)

        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate V_phi(s) without adding the estimate to an autograd graph."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return float(self.critic(state_tensor).item())

    def n_step_returns(
        self,
        rewards: list[float],
        bootstrap_value: float,
    ) -> torch.Tensor:
        """Calculate returns ending in a bootstrapped critic estimate."""
        returns = []
        return_from_t = bootstrap_value

        # For the final rollout state, bootstrap_value is either V(S_end) or 0
        # at a true terminal state. Work backward through observed rewards.
        for reward in reversed(rewards):
            return_from_t = reward + self.gamma * return_from_t
            returns.append(return_from_t)

        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def update(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        bootstrap_value: float,
    ) -> tuple[float, float, float]:
        """Update actor and critic once from one short rollout."""
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        returns = self.n_step_returns(rewards, bootstrap_value)

        distribution = Categorical(logits=self.actor(state_tensor))
        selected_log_probabilities = distribution.log_prob(action_tensor)
        entropy = distribution.entropy().mean()
        value_estimates = self.critic(state_tensor)

        # The n-step TD error is an advantage estimate for the actor.
        advantages = returns - value_estimates

        # detach() prevents the actor loss from changing the critic. A small
        # entropy bonus discourages the stochastic policy from collapsing too
        # early to a single action.
        actor_loss = -(
            selected_log_probabilities * advantages.detach()
        ).mean() - self.entropy_coefficient * entropy

        # Huber loss is less sensitive than MSE to occasional large TD targets.
        critic_loss = F.smooth_l1_loss(value_estimates, returns)

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
            float(advantages.detach().mean().item()),
        )


def train(
    agent: ActorCriticAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
    rollout_steps: int,
) -> tuple[list[float], list[float], list[float]]:
    """Update after each rollout, potentially several times in one episode."""
    episode_returns: list[float] = []
    mean_actor_losses: list[float] = []
    mean_critic_losses: list[float] = []

    for episode in range(episodes):
        episode_seed = seed if episode == 0 else None
        state, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        episode_return = 0.0
        actor_losses = []
        critic_losses = []
        advantages = []

        while not (terminated or truncated):
            rollout_states: list[np.ndarray] = []
            rollout_actions: list[int] = []
            rollout_rewards: list[float] = []

            # Collect at most n transitions. Unlike REINFORCE, Actor-Critic does
            # not have to wait for the complete episode before learning.
            for _ in range(rollout_steps):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                rollout_states.append(state)
                rollout_actions.append(action)
                rollout_rewards.append(float(reward))
                episode_return += float(reward)
                state = next_state

                if terminated or truncated:
                    break

            # A true terminal state has zero future value. A time-limit
            # truncation is not a task terminal, so bootstrap V(S_end) there.
            bootstrap_value = (
                0.0 if terminated else agent.estimate_value(state)
            )

            actor_loss, critic_loss, mean_advantage = agent.update(
                rollout_states,
                rollout_actions,
                rollout_rewards,
                bootstrap_value,
            )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            advantages.append(mean_advantage)

        episode_returns.append(episode_return)
        mean_actor_losses.append(float(np.mean(actor_losses)))
        mean_critic_losses.append(float(np.mean(critic_losses)))

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"回合：{episode + 1:4d}，总奖励：{episode_return:5.1f}，"
                f"策略损失：{mean_actor_losses[-1]:8.4f}，"
                f"价值损失：{mean_critic_losses[-1]:8.4f}，"
                f"平均优势：{np.mean(advantages):7.3f}"
            )

    recent_count = min(100, len(episode_returns))
    recent_average = float(np.mean(episode_returns[-recent_count:]))
    print(f"最近 {recent_count} 回合的平均奖励：{recent_average:.1f}")
    return episode_returns, mean_actor_losses, mean_critic_losses


def plot_training(
    episode_returns: list[float],
    actor_losses: list[float],
    critic_losses: list[float],
    output_path: Path,
) -> None:
    """Save episode returns and mean per-episode actor/critic losses."""
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
    axes[0].set_title("n-step Actor-Critic on CartPole-v1")
    axes[1].plot(episode_numbers, actor_losses)
    axes[1].set_ylabel("Actor loss")
    axes[2].plot(episode_numbers, critic_losses)
    axes[2].set(xlabel="Episode", ylabel="Critic loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def evaluate(agent: ActorCriticAgent, seed: int) -> list[np.ndarray]:
    """Run one greedy test episode and collect RGB frames."""
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
    """Save a greedy test episode as GIF and MP4 animations."""
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
    result.save(output_dir / "code-6-actor-critic-cartpole.mp4", writer="ffmpeg")
    result.save(output_dir / "code-6-actor-critic-cartpole.gif", writer="pillow")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=20)
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
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    training_env = gym.make("CartPole-v1")
    training_env.action_space.seed(args.seed)
    agent = ActorCriticAgent(
        state_size=training_env.observation_space.shape[0],
        action_size=training_env.action_space.n,
    )

    try:
        episode_returns, actor_losses, critic_losses = train(
            agent,
            training_env,
            args.episodes,
            args.seed,
            args.rollout_steps,
        )
    finally:
        training_env.close()

    plot_training(
        episode_returns,
        actor_losses,
        critic_losses,
        args.output_dir / "code-6-actor-critic-training.pdf",
    )

    if not args.no_animation:
        frames = evaluate(agent, seed=args.seed + 1)
        save_animation(frames, args.output_dir)


if __name__ == "__main__":
    main()

#  The example implements stable n-step Actor–Critic:
#
#   - Actor learns (\pi_\theta(a\mid s)).
#   - Critic learns (V_\phi(s)).
#   - Collects rollouts of up to 20 transitions.
#   - Uses bootstrapped n-step returns.
#   - Updates several times during an episode.
#   - Uses detached advantages, an entropy bonus, Huber critic loss, and gradient clipping.
#   - Handles termination and time-limit truncation correctly.

#   uv run python rl_learning_demo/day02/code-6-actor-Critic.py
#
#   Shorter run:
#
#   uv run python rl_learning_demo/day02/code-6-actor-Critic.py \
#       --episodes 500 \
#       --no-animation