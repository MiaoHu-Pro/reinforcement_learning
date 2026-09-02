"""Train CartPole-v1 with batched one-step TD Actor-Critic.

This is the Gymnasium/PyTorch rewrite of ``code-2-old.py``.

The learning signal is still the same one-step TD error used by Code 1:

    delta_t = R_t + gamma * V(S_(t+1)) - V(S_t)

The difference is *when* learning happens:

* Code 1 updates the actor and critic immediately after every transition.
* Code 2 first collects one complete trajectory, calculates every one-step TD
  error as a vectorized batch, and updates both networks once per episode.

Collecting a complete trajectory is not required by one-step TD theory. It is
an intentional batching choice that keeps one policy fixed during collection,
averages gradients over many transitions, and prepares the workflow used by
multi-step methods such as GAE.


 The main difference is update timing:

   Feature                         Code 1                         Code 2
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TD error                        One-step                       One-step
  ──────────────────────────────  ─────────────────────────────  ────────────────────────────────
   Data collected before update    One transition                 One complete trajectory
  ──────────────────────────────  ─────────────────────────────  ────────────────────────────────
   Updates per episode             One per transition             One
  ──────────────────────────────  ─────────────────────────────  ────────────────────────────────
   Policy during episode           May change after every step    Fixed throughout collection
  ──────────────────────────────  ─────────────────────────────  ────────────────────────────────
   Gradient                        From one transition            Averaged across the trajectory
  ──────────────────────────────  ─────────────────────────────  ────────────────────────────────
   Memory                          Minimal                        Stores the episode

  Code 2 does not mathematically need a complete trajectory to calculate a one-step TD error. Trajectory collection is an intentional batching strategy that:

  - processes all transitions efficiently as tensors;
  - averages noisy transition gradients;
  - avoids updating the policy during collection;
  - reduces the number of optimizer calls;
  - establishes the workflow needed by GAE and other multi-step methods.

  However, it delays learning until the episode ends and requires more memory.
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


# One trajectory contains all transitions from S_0 until this episode ends.
Trajectory = tuple[
    list[np.ndarray],  # states:       S_0, ..., S_(T-1)
    list[np.ndarray],  # next_states:  S_1, ..., S_T
    list[int],         # actions:      A_0, ..., A_(T-1)
    list[float],       # rewards:      R_0, ..., R_(T-1)
    list[bool],        # true termination flags for each transition
]


class ActorNet(nn.Module):
    """Represent the stochastic policy pi_theta(a | s)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(states))
        # Return logits. log_softmax or Categorical can convert them into
        # stable log probabilities without taking log(softmax(...)) manually.
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


class TrajectoryBatchActorCriticAgent:
    """Make one batched update from one complete on-policy trajectory."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.entropy_coefficient = 0.001

        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0002)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

    def choose_action(self, state: np.ndarray) -> int:
        """Sample an action without retaining a per-step autograd graph.

        Code 1 must retain each selected action's log-probability because it
        calls backward() immediately. Code 2 stores states and actions instead.
        It reconstructs all log-probabilities together during the batch update.
        This avoids storing an autograd graph throughout the entire episode.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(state_tensor))
            return int(distribution.sample().item())

    def collect_trajectory(
        self,
        env: gym.Env,
        seed: int | None = None,
    ) -> Trajectory:
        """Collect one complete episode before changing network weights.

        Why collect it?

        1. Every action in the trajectory comes from one unchanged policy.
        2. All transitions can be processed efficiently as tensor batches.
        3. Many noisy transition gradients are averaged into one update.
        4. The stored sequence can later support GAE or n-step returns.

        The cost is delayed learning and memory proportional to episode length.
        Again, collection is a batching decision, not a one-step TD necessity.
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

            # Store the transition (S_t, A_t, R_t, S_(t+1)). No backward pass
            # and no optimizer step occurs inside this collection loop.
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(float(reward))

            # A true termination has no future value. A time-limit truncation
            # ends collection but should still bootstrap from V(S_(t+1)).
            terminations.append(bool(terminated))
            state = next_state

        return states, next_states, actions, rewards, terminations

    def update(self, trajectory: Trajectory) -> tuple[float, float, float]:
        """Calculate all one-step TD errors and update once.

        If the trajectory contains T transitions, this method calculates T TD
        targets and T TD errors simultaneously. It then calls backward() once
        for the actor and once for the critic—not T times.
        """
        states, next_states, actions, rewards, terminations = trajectory

        # Shapes after conversion:
        # states and next_states: [T, state_size]
        # actions/rewards/terminations: [T]
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
        )
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminations, dtype=torch.float32)

        # V_phi(S_0), V_phi(S_1), ..., V_phi(S_(T-1))
        # This evaluation needs gradients because it trains the critic.
        values = self.critic(state_tensor)

        # Build fixed one-step TD targets. There must be no gradient through
        # V_phi(S_(t+1)); this is a semi-gradient TD update.
        with torch.no_grad():
            next_values = self.critic(next_state_tensor)
            td_targets = (
                reward_tensor
                + self.gamma
                * (1.0 - terminated_tensor)
                * next_values
            )

        # One separate one-step error is calculated for every transition:
        # delta_t = R_t + gamma*V_phi(S_(t+1)) - V_phi(S_t).
        # These errors are a vector; they are not recursively combined as GAE.
        td_errors = td_targets - values

        # Train V_phi(S_t) toward its detached one-step TD target. MSE performs
        # a mean across the T transitions, giving one averaged critic update.
        critic_loss = F.mse_loss(values, td_targets)

        # Re-evaluate the complete trajectory with the unchanged actor. Using
        # log_softmax followed by gather demonstrates how a batch selects
        # log pi_theta(A_t | S_t) for the action actually taken at every t.
        action_logits = self.actor(state_tensor)              # [T, 2]
        all_log_probabilities = F.log_softmax(
            action_logits,
            dim=1,
        )                                                     # [T, 2]
        selected_log_probabilities = all_log_probabilities.gather(
            dim=1,
            index=action_tensor.unsqueeze(1),                 # [T, 1]
        ).squeeze(1)                                          # [T]

        # Example of gather:
        # all_log_probabilities = [[log P(A=0|S_0), log P(A=1|S_0)],
        #                          [log P(A=0|S_1), log P(A=1|S_1)]]
        # actions = [1, 0]
        # selected result = [log P(A=1|S_0), log P(A=0|S_1)]

        # The TD error estimates the sampled action's advantage. detach() makes
        # it a constant actor learning signal, so actor_loss cannot update the
        # critic. mean() avoids making the gradient scale grow with episode
        # length; code-2-old.py used sum(), whose scale changed with T.
        distribution = Categorical(logits=action_logits)
        entropy = distribution.entropy().mean()
        actor_loss = -(
            selected_log_probabilities * td_errors.detach()
        ).mean() - self.entropy_coefficient * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # backward() calculates gradients. It does not change weights.
        actor_loss.backward()
        critic_loss.backward()

        # These two calls perform the only weight changes for this episode.
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(td_errors.detach().abs().mean().item()),
        )


def train(
    agent: TrajectoryBatchActorCriticAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Alternate between one complete collection and one batch update."""
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(episodes):
        # COLLECTION PHASE: network weights remain fixed.
        trajectory = agent.collect_trajectory(env, seed=seed + episode)

        # UPDATE PHASE: all stored transitions contribute to one gradient.
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
    """Save episode returns and batch losses without opening a GUI."""
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
        title="Trajectory-batched one-step TD Actor-Critic",
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
        description=(
            "Train trajectory-batched one-step TD Actor-Critic on CartPole-v1."
        )
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "actor-critic-trajectory-batch-one-step-td.pdf"
        ),
        help="Path of the training plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = TrajectoryBatchActorCriticAgent(state_size, action_size)

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
