"""Train CartPole-v1 with trajectory-batched Actor-Critic and GAE.

This program extends
``code-2-actor-critic-trajectory-batch-one-step-td.py``.

Both programs collect one fresh on-policy trajectory and update once after the
episode. The difference is the learning target:

* one-step version: actor advantage = delta_t
* this GAE version: actor advantage = delta_t + gamma*lambda*A_(t+1)

The GAE advantage also constructs a multi-step critic target:

    value_target_t = A_t^GAE + V_old(S_t)

All old values, TD errors, advantages, and targets are detached before either
optimizer changes the network weights.
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
    list[np.ndarray],  # states:       S_0, ..., S_(T-1)
    list[np.ndarray],  # next_states:  S_1, ..., S_T
    list[int],         # actions:      A_0, ..., A_(T-1)
    list[float],       # rewards:      R_0, ..., R_(T-1)
    list[bool],        # true task termination flags
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


class TrajectoryGAEAgent:
    """Update an actor and critic once from one trajectory and its GAE."""

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
        """Sample from the current policy without storing an autograd graph."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(state_tensor))
            return int(distribution.sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect one new complete trajectory using unchanged policy weights.

        GAE needs an ordered sequence of TD errors. Therefore the transitions
        are stored in temporal order rather than updated and discarded online.
        No backward pass or optimizer step occurs during collection.
        """
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminations: list[bool] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(float(reward))
            # Only true termination removes the next-state bootstrap value.
            # A time-limit truncation stops collection but is not terminal.
            terminations.append(bool(terminated))
            state = next_state

        return states, next_states, actions, rewards, terminations

    def calculate_gae(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        old_values: torch.Tensor,
        old_next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one-step errors, GAE advantages, and critic targets."""
        # First calculate every ordinary one-step TD target:
        # y_t = R_t + gamma*(1-terminated_t)*V_old(S_(t+1)).
        one_step_td_targets = (
            rewards
            + self.gamma
            * (1.0 - terminations)
            * old_next_values
        )

        # delta_t = y_t - V_old(S_t)
        td_errors = one_step_td_targets - old_values

        # Work backward because A_t depends on A_(t+1):
        # A_t^GAE = delta_t
        #             + gamma*lambda*(1-terminated_t)*A_(t+1)^GAE.
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

        # With lambda=0, this reduces exactly to the one-step TD target:
        # delta_t + V_old(S_t) = y_t.
        # With lambda near 1, it becomes a longer-horizon return-like target.
        value_targets = gae_advantages + old_values
        return td_errors, gae_advantages, value_targets

    def update(self, trajectory: Trajectory) -> tuple[float, float, float]:
        """Calculate trajectory GAE, then update actor and critic once."""
        states, next_states, actions, rewards, terminations = trajectory

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
        )
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminations, dtype=torch.float32)

        # Snapshot the critic before either optimizer step. These old values
        # make all targets fixed constants during the coming update.
        with torch.no_grad():
            old_values = self.critic(state_tensor)
            old_next_values = self.critic(next_state_tensor)
            td_errors, gae_advantages, value_targets = self.calculate_gae(
                reward_tensor,
                terminated_tensor,
                old_values,
                old_next_values,
            )

        # Rebuild the actor graph for every stored state. gather() selects the
        # log-probability of the action actually sampled at each time step.
        action_logits = self.actor(state_tensor)              # [T, actions]
        all_log_probabilities = F.log_softmax(
            action_logits,
            dim=1,
        )                                                     # [T, actions]
        selected_log_probabilities = all_log_probabilities.gather(
            dim=1,
            index=action_tensor.unsqueeze(1),                 # [T, 1]
        ).squeeze(1)                                          # [T]

        distribution = Categorical(logits=action_logits)
        entropy = distribution.entropy().mean()

        # Standardization stabilizes only the scale of the actor gradient. It
        # must not be used for the critic, whose targets need the value scale.
        actor_advantages = gae_advantages
        if actor_advantages.numel() > 1:
            actor_advantages = (
                actor_advantages - actor_advantages.mean()
            ) / (actor_advantages.std(unbiased=False) + 1e-8)

        actor_loss = -(
            selected_log_probabilities * actor_advantages
        ).mean() - self.entropy_coefficient * entropy

        # Evaluate V_phi(S_t) again with gradients enabled. The target on the
        # right is detached because it was constructed under no_grad().
        values = self.critic(state_tensor)
        critic_loss = F.mse_loss(values, value_targets)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # backward() calculates gradients for one complete trajectory batch.
        actor_loss.backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        # These are the only two weight changes for the current episode.
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(td_errors.abs().mean().item()),
        )


def train(
    agent: TrajectoryGAEAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Repeatedly collect a new trajectory and perform one GAE update."""
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(episodes):
        # Phase 1: collect fresh on-policy experience without updating.
        trajectory = agent.collect_trajectory(env, seed=seed + episode)

        # Phase 2: use that trajectory once, then discard it. The next episode
        # collects a new trajectory with the newly updated policy.
        actor_loss, critic_loss, mean_absolute_td_error = agent.update(
            trajectory
        )
        episode_return = float(sum(trajectory[3]))

        episode_returns.append(episode_return)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"steps {len(trajectory[3]):3d} | "
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
    """Save episode returns and losses without opening an interactive GUI."""
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

    axes[0].set(
        ylabel="Return",
        title="Trajectory-batched Actor-Critic with GAE",
    )
    axes[1].plot(episode_numbers, actor_losses)
    axes[1].set_ylabel("Actor loss")
    axes[2].plot(episode_numbers, critic_losses)
    axes[2].set(xlabel="Episode", ylabel="Critic loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train trajectory-batched Actor-Critic with GAE."
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
        default=Path(__file__).with_name(
            "actor-critic-trajectory-GAE-training.pdf"
        ),
        help="Path of the training plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = TrajectoryGAEAgent(
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
