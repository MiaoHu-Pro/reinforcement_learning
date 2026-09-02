"""Train CartPole-v1 with Actor-Critic and GAE value targets.

The two one-step Actor-Critic examples in this folder use only

    delta_t = R_t + gamma * V_old(S_(t+1)) - V_old(S_t)

for transition t.  This example combines the current and future one-step TD
errors with Generalized Advantage Estimation (GAE):

    A_hat_t = delta_t + gamma*lambda*A_hat_(t+1)

The same detached GAE estimate trains both networks in different ways:

    actor advantage = A_hat_t
    critic target   = A_hat_t + V_old(S_t)

One complete on-policy episode is collected before each batched update.  This
keeps the relationship among TD errors, GAE advantages, and critic targets
visible for learning purposes.
"""

import argparse
from pathlib import Path

import gymnasium as gym
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
    list[bool],        # true task termination at transition t
]


class ActorNet(nn.Module):
    """Represent the stochastic policy pi_theta(a | s)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden)  # action logits


class CriticNet(nn.Module):
    """Approximate the state-value function V_phi(s)."""

    def __init__(self, state_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden).squeeze(-1)


class GAEActorCriticAgent:
    """Use a weighted sequence of TD errors for actor and critic updates."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        gae_lambda: float,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 0.001

        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def choose_action(self, state: np.ndarray) -> int:
        """Sample an action from the current on-policy distribution."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(state_tensor))
            return int(distribution.sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect exactly one complete episode with the current policy."""
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
            # Store only true termination here. A time-limit truncation should
            # still bootstrap from V(S_(t+1)).
            terminations.append(bool(terminated))
            state = next_state

        return states, actions, rewards, next_states, terminations

    def calculate_gae_and_value_targets(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        old_values: torch.Tensor,
        old_next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate TD errors, GAE advantages, and GAE value targets.

        All input value estimates are old, detached values measured before the
        optimizer update. Consequently, every returned tensor is a fixed
        learning target and has no autograd graph.
        """
        # y_t^(1) = R_t + gamma * (1 - terminated_t) * V_old(S_(t+1))
        one_step_td_targets = (
            rewards
            + self.gamma * (1.0 - terminations) * old_next_values
        )

        # delta_t = y_t^(1) - V_old(S_t)
        td_errors = one_step_td_targets - old_values

        # A_hat_t^GAE = delta_t + gamma*lambda*A_hat_(t+1)^GAE
        #
        # Because one complete episode is processed at a time, the backward
        # recursion begins with zero beyond its final collected transition.
        advantages_reversed: list[torch.Tensor] = []
        next_advantage = torch.zeros((), dtype=torch.float32)
        for index in range(len(rewards) - 1, -1, -1):
            continuation_mask = 1.0 - terminations[index]
            next_advantage = (
                td_errors[index]
                + self.gamma
                * self.gae_lambda
                * continuation_mask
                * next_advantage
            )
            advantages_reversed.append(next_advantage)

        gae_advantages = torch.stack(list(reversed(advantages_reversed)))

        # The requested critic target:
        # V_hat_t^target = A_hat_t^GAE + V_old(S_t)
        #
        # Neither term has a gradient. The *new* V_phi(S_t), evaluated later,
        # is optimized toward this fixed target.
        value_targets = gae_advantages + old_values
        return td_errors, gae_advantages, value_targets

    def update(self, trajectory: Trajectory) -> tuple[float, float, float]:
        """Perform one batched actor update and one batched critic update."""
        states, actions, rewards, next_states, terminations = trajectory

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
        )
        terminated_tensor = torch.tensor(terminations, dtype=torch.float32)

        # Snapshot the critic before either optimizer changes any weights.
        # These values implement stop-gradient / old-value handling.
        with torch.no_grad():
            old_values = self.critic(state_tensor)
            old_next_values = self.critic(next_state_tensor)
            td_errors, gae_advantages, value_targets = (
                self.calculate_gae_and_value_targets(
                    reward_tensor,
                    terminated_tensor,
                    old_values,
                    old_next_values,
                )
            )

        # Re-evaluate both networks with gradients enabled. selected_log_probs
        # belongs only to the actor graph; values belongs only to the critic.
        distribution = Categorical(logits=self.actor(state_tensor))
        selected_log_probabilities = distribution.log_prob(action_tensor)
        entropy = distribution.entropy().mean()
        values = self.critic(state_tensor)

        # Standardization changes only the scale of the actor learning signal.
        # The critic must use the original, unstandardized value targets.
        actor_advantages = gae_advantages
        if actor_advantages.numel() > 1:
            actor_advantages = (
                actor_advantages - actor_advantages.mean()
            ) / (actor_advantages.std(unbiased=False) + 1e-8)

        actor_loss = -(
            selected_log_probabilities * actor_advantages
        ).mean() - self.entropy_coefficient * entropy

        # L_critic = (1/N) * sum_t (V_phi(S_t) - V_hat_t^target)^2
        critic_loss = F.mse_loss(values, value_targets)

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
            float(td_errors.abs().mean().item()),
        )


def train(
    agent: GAEActorCriticAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Collect one episode and then update once, repeatedly."""
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(episodes):
        trajectory = agent.collect_trajectory(env, seed=seed + episode)

        # The whole episode is now available. GAE can propagate later TD
        # errors backward to earlier state-action pairs before backward().
        actor_loss, critic_loss, mean_absolute_td_error = agent.update(
            trajectory
        )
        episode_return = float(sum(trajectory[2]))

        episode_returns.append(episode_return)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"return {episode_return:5.1f} | "
                f"actor loss {actor_loss:8.4f} | "
                f"critic loss {critic_loss:8.4f} | "
                f"mean |TD error| {mean_absolute_td_error:7.3f}"
            )

    recent_count = min(100, len(episode_returns))
    print(
        f"Mean return over the last {recent_count} episodes: "
        f"{np.mean(episode_returns[-recent_count:]):.1f}"
    )
    return episode_returns, actor_losses, critic_losses


def plot_training(
    episode_returns: list[float],
    actor_losses: list[float],
    critic_losses: list[float],
    output_path: Path,
) -> None:
    """Save returns and losses without opening an interactive window."""
    episode_numbers = np.arange(1, len(episode_returns) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(episode_numbers, episode_returns, alpha=0.35)
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

    axes[0].set(ylabel="Return", title="Actor-Critic with GAE on CartPole-v1")
    axes[1].plot(episode_numbers, actor_losses)
    axes[1].set_ylabel("Actor loss")
    axes[2].plot(episode_numbers, critic_losses)
    axes[2].set(xlabel="Episode", ylabel="Critic loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Actor-Critic with GAE on CartPole-v1."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE bias-variance parameter in [0, 1].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("actor-critic-GAE-training.pdf"),
        help="Path of the training plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be between 0 and 1")
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = GAEActorCriticAgent(
        state_size,
        action_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    try:
        episode_returns, actor_losses, critic_losses = train(
            agent,
            env,
            args.episodes,
            args.seed,
        )
    finally:
        env.close()

    plot_training(
        episode_returns,
        actor_losses,
        critic_losses,
        args.output,
    )
    print(f"Saved training plot to: {args.output}")


if __name__ == "__main__":
    main()
