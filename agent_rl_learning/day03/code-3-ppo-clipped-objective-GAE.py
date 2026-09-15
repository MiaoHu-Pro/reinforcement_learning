"""Train CartPole-v1 with Proximal Policy Optimization (PPO-Clip).

This is the Gymnasium/PyTorch rewrite of ``code-3-old.py``.

PPO begins with the same actor, critic, and GAE ideas as the preceding Day 3
Actor-Critic examples. Its important new idea is safe data reuse. After one
trajectory is collected with the old policy, PPO performs several optimization
epochs on that same data. The clipped probability-ratio objective discourages
any one update from moving the new policy too far from the data-generating
policy:

    ratio_t = pi_theta(A_t | S_t) / pi_old(A_t | S_t)

    L_clip = -mean(min(
        ratio_t * A_t,
        clip(ratio_t, 1-epsilon, 1+epsilon) * A_t,
    ))

The trajectory is discarded after the PPO update. The next episode collects a
fresh trajectory with the newly updated policy.
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
    list[np.ndarray],  # states:                S_0, ..., S_(T-1)
    list[np.ndarray],  # next states:           S_1, ..., S_T
    list[int],         # sampled actions:       A_0, ..., A_(T-1)
    list[float],       # rewards:               R_0, ..., R_(T-1)
    list[bool],        # true termination flags
    list[float],       # log pi_old(A_t | S_t)
]


class ActorNet(nn.Module):
    """Represent the stochastic policy pi_theta(a | s)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        # Returning logits lets Categorical calculate stable probabilities and
        # log probabilities without explicitly computing log(softmax(...)).
        return self.output(hidden)


class CriticNet(nn.Module):
    """Approximate the state-value function V_phi(s)."""

    def __init__(self, state_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        return self.output(hidden).squeeze(-1)


class PPOAgent:
    """Train an actor and critic with PPO's clipped surrogate objective."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        update_epochs: int,
        minibatch_size: int,
        target_kl: float,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.entropy_coefficient = 0.001

        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def choose_action(self, state: np.ndarray) -> tuple[int, float]:
        """Sample A_t and save its old-policy log probability.

        PPO must remember log pi_old(A_t | S_t). During repeated updates, the
        current actor changes but this saved value remains fixed and forms the
        denominator of the PPO probability ratio.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            old_distribution = Categorical(logits=self.actor(state_tensor))
            action_tensor = old_distribution.sample()
            old_log_probability = old_distribution.log_prob(action_tensor)

        return int(action_tensor.item()), float(old_log_probability.item())

    def collect_trajectory(
        self,
        env: gym.Env,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect one fresh on-policy episode without changing weights.

        The actor must remain fixed while collecting this batch. Otherwise the
        saved ``old_log_probabilities`` would come from multiple old policies,
        making the PPO ratio harder to interpret.
        """
        state, _ = env.reset(seed=seed)
        states: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminations: list[bool] = []
        old_log_probabilities: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action, old_log_probability = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(float(reward))
            # A true terminal state has zero future value. A time-limit
            # truncation ends this rollout but still permits bootstrapping.
            terminations.append(bool(terminated))
            old_log_probabilities.append(old_log_probability)
            state = next_state

        return (
            states,
            next_states,
            actions,
            rewards,
            terminations,
            old_log_probabilities,
        )

    def calculate_gae_and_value_targets(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        old_values: torch.Tensor,
        old_next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build detached GAE advantages and lambda-return value targets."""
        # One-step semi-gradient TD targets and residuals:
        # y_t = R_t + gamma*(1-terminated_t)*V_old(S_(t+1))
        # delta_t = y_t - V_old(S_t)
        one_step_td_targets = (
            rewards
            + self.gamma
            * (1.0 - terminations)
            * old_next_values
        )
        td_errors = one_step_td_targets - old_values

        # Generalized Advantage Estimation runs backward through the ordered
        # trajectory and combines current and future one-step TD errors:
        # A_t^GAE = delta_t
        #           + gamma*lambda*(1-terminated_t)*A_(t+1)^GAE.
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

        # The critic learns from an unnormalized return-like target:
        # V_target_t = A_t^GAE + V_old(S_t).
        value_targets = gae_advantages + old_values
        return td_errors, gae_advantages, value_targets

    def update(
        self,
        trajectory: Trajectory,
    ) -> tuple[float, float, float, float, int]:
        """Reuse one trajectory for several PPO minibatch epochs.

        Returns mean actor loss, critic loss, approximate KL divergence, clip
        fraction, and the number of optimization epochs actually completed.
        """
        (
            states,
            next_states,
            actions,
            rewards,
            terminations,
            old_log_probabilities,
        ) = trajectory

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
        )
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminations, dtype=torch.float32)
        old_log_probability_tensor = torch.tensor(
            old_log_probabilities,
            dtype=torch.float32,
        )

        # Freeze all rollout targets before updating. These tensors describe
        # the behavior of the old actor/critic that generated the trajectory.
        with torch.no_grad():
            old_values = self.critic(state_tensor)
            old_next_values = self.critic(next_state_tensor)
            _, gae_advantages, value_targets = (
                self.calculate_gae_and_value_targets(
                    reward_tensor,
                    terminated_tensor,
                    old_values,
                    old_next_values,
                )
            )

        # Advantage normalization changes the actor signal's scale, not its
        # ordering. The critic still uses unnormalized ``value_targets``.
        normalized_advantages = gae_advantages
        if normalized_advantages.numel() > 1:
            normalized_advantages = (
                normalized_advantages - normalized_advantages.mean()
            ) / (normalized_advantages.std(unbiased=False) + 1e-8)

        transition_count = len(states)
        effective_minibatch_size = min(
            self.minibatch_size,
            transition_count,
        )
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        completed_epochs = 0

        # Unlike the preceding Actor-Critic examples, PPO deliberately reuses
        # the same trajectory. Each epoch shuffles it into new minibatches.
        for _ in range(self.update_epochs):
            shuffled_indices = torch.randperm(transition_count)
            epoch_kls: list[float] = []

            for start in range(0, transition_count, effective_minibatch_size):
                indices = shuffled_indices[
                    start : start + effective_minibatch_size
                ]

                batch_states = state_tensor[indices]
                batch_actions = action_tensor[indices]
                batch_old_log_probabilities = old_log_probability_tensor[
                    indices
                ]
                batch_advantages = normalized_advantages[indices]
                batch_value_targets = value_targets[indices]

                # Recalculate log pi_theta(A_t|S_t) with the current actor.
                # It changes after every optimizer step; old_log_probability
                # remains frozen for the entire PPO update.
                distribution = Categorical(
                    logits=self.actor(batch_states)
                )
                new_log_probabilities = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                # Computing the ratio in log space is numerically stable:
                # exp(log pi_new - log pi_old) = pi_new / pi_old.
                log_ratios = (
                    new_log_probabilities
                    - batch_old_log_probabilities
                )
                probability_ratios = torch.exp(log_ratios)

                # Unclipped and clipped surrogate objectives. Taking the
                # minimum creates a pessimistic lower bound on improvement and
                # removes the incentive to push an already-large ratio farther.
                unclipped_surrogate = (
                    probability_ratios * batch_advantages
                )
                clipped_ratios = torch.clamp(
                    probability_ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )
                clipped_surrogate = clipped_ratios * batch_advantages

                actor_loss = -torch.minimum(
                    unclipped_surrogate,
                    clipped_surrogate,
                ).mean() - self.entropy_coefficient * entropy

                # The critic is also reused for several epochs, but every
                # prediction is trained toward the same detached GAE target.
                value_predictions = self.critic(batch_states)
                critic_loss = F.mse_loss(
                    value_predictions,
                    batch_value_targets,
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # backward() calculates gradients; step() changes weights.
                actor_loss.backward()
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=0.5,
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Diagnostics are not part of either loss. Approximate KL
                # measures policy movement; clip fraction reports how many
                # samples lie outside the allowed ratio interval.
                with torch.no_grad():
                    approximate_kl = (
                        (probability_ratios - 1.0) - log_ratios
                    ).mean()
                    clip_fraction = (
                        (probability_ratios - 1.0).abs()
                        > self.clip_epsilon
                    ).float().mean()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                approximate_kls.append(float(approximate_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))
                epoch_kls.append(float(approximate_kl.item()))

            completed_epochs += 1

            # Clipping discourages overly large movement but does not impose a
            # hard distance limit. Early stopping adds a second safeguard.
            if np.mean(epoch_kls) > self.target_kl:
                break

        return (
            float(np.mean(actor_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(approximate_kls)),
            float(np.mean(clip_fractions)),
            completed_epochs,
        )


def train(
    agent: PPOAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Alternate between fresh data collection and one PPO update."""
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(episodes):
        # Phase 1: collect with fixed old policy parameters.
        trajectory = agent.collect_trajectory(env, seed=seed + episode)

        # Phase 2: reuse this trajectory for clipped PPO optimization.
        (
            actor_loss,
            critic_loss,
            approximate_kl,
            clip_fraction,
            completed_epochs,
        ) = agent.update(trajectory)

        episode_return = float(sum(trajectory[3]))
        episode_returns.append(episode_return)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode == 0 or (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"steps {len(trajectory[3]):3d} | "
                f"return {episode_return:5.1f} | "
                f"actor loss {actor_loss:8.4f} | "
                f"critic loss {critic_loss:8.4f} | "
                f"KL {approximate_kl:7.5f} | "
                f"clip {clip_fraction:5.1%} | "
                f"epochs {completed_epochs:2d}"
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
    """Save returns and PPO losses without opening an interactive GUI."""
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

    axes[0].set(ylabel="Return", title="PPO-Clip with GAE on CartPole-v1")
    axes[1].plot(episode_numbers, actor_losses)
    axes[1].set_ylabel("Mean actor loss")
    axes[2].plot(episode_numbers, critic_losses)
    axes[2].set(xlabel="Episode", ylabel="Mean critic loss")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO-Clip with GAE on CartPole-v1."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO ratio clipping distance; 0.2 gives [0.8, 1.2].",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="Maximum passes over each collected trajectory.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Transitions per optimizer step; capped by trajectory length.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Stop PPO epochs early when mean approximate KL exceeds this.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ppo-clipped-objective-GAE.pdf"),
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
    if not 0.0 < args.clip_epsilon < 1.0:
        raise ValueError("--clip-epsilon must be between 0 and 1")
    if args.update_epochs < 1:
        raise ValueError("--update-epochs must be at least 1")
    if args.minibatch_size < 1:
        raise ValueError("--minibatch-size must be at least 1")
    if args.target_kl <= 0.0:
        raise ValueError("--target-kl must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = PPOAgent(
        state_size,
        action_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
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
