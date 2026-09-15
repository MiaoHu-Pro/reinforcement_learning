"""Train CartPole-v1 with online one-step Actor-Critic.

This is the Gymnasium/PyTorch rewrite of ``code-1-old.py``.  The algorithm
updates the actor and critic after every transition rather than waiting for a
complete trajectory:

    delta_t = R_t + gamma * V(S_(t+1)) - V(S_t)

The critic minimizes the squared TD error.  The actor uses the detached TD
error as an estimate of the advantage of the sampled action.
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


class ActorNet(nn.Module):
    """Represent the stochastic policy pi_theta(a | s)."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(state))
        # Categorical(logits=...) applies the required softmax internally.
        # Returning logits is more numerically stable than returning softmax
        # probabilities and then applying log() to a selected probability.
        return self.output(hidden)


class CriticNet(nn.Module):
    """Approximate the state-value function V_phi(s)."""

    def __init__(self, state_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(state))
        return self.output(hidden).squeeze(-1)


class OnlineActorCriticAgent:
    """Update an actor and critic from each individual transition."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0002)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

    def choose_action(
        self,
        state: np.ndarray,
    ) -> tuple[int, torch.Tensor]:
        """Sample A_t and retain log pi_theta(A_t | S_t) for learning."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        distribution = Categorical(logits=self.actor(state_tensor))
        action_tensor = distribution.sample()
        log_probability = distribution.log_prob(action_tensor)
        return int(action_tensor.item()), log_probability

    def update(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        action_log_probability: torch.Tensor,
        terminated: bool,
    ) -> tuple[float, float, float]:
        """Perform one actor update and one critic update.

        ``terminated`` controls bootstrapping.  A true terminal CartPole state
        has no future value.  A Gymnasium time-limit truncation is different:
        the task itself has not terminated, so V(S_(t+1)) is still used.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)

        value = self.critic(state_tensor)

        # The TD target is treated as a fixed training label.  no_grad() is the
        # effective version of the old file's unused ``target.detach()`` call.
        with torch.no_grad():
            next_value = (
                torch.zeros((), dtype=torch.float32)
                if terminated
                else self.critic(next_state_tensor)
            )
            td_target = torch.as_tensor(reward, dtype=torch.float32)

            td_target = td_target + self.gamma * next_value

        # delta_t tells the critic how wrong V(S_t) was and tells the actor
        # whether the sampled action performed better or worse than expected.
        td_error = td_target - value
        critic_loss = F.mse_loss(value, td_target)

        # detach() keeps the actor update from backpropagating into the critic.
        actor_loss = -action_log_probability * td_error.detach()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(td_error.detach().item()),
        )


def train(
    agent: OnlineActorCriticAgent,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> list[float]:
    """Train online: step once, then immediately update both networks."""
    episode_returns: list[float] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_return = 0.0
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        td_errors: list[float] = []

        while not (terminated or truncated):
            action, action_log_probability = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # This is the defining workflow of this example: one environment
            # transition is immediately followed by backward() and step().
            actor_loss, critic_loss, td_error = agent.update(
                state,
                next_state,
                float(reward),
                action_log_probability,
                terminated,
            )

            state = next_state
            episode_return += float(reward)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            td_errors.append(td_error)

        episode_returns.append(episode_return)

        if episode == 0 or (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"return {episode_return:5.1f} | "
                f"actor loss {np.mean(actor_losses):8.4f} | "
                f"critic loss {np.mean(critic_losses):8.4f} | "
                f"mean TD error {np.mean(td_errors):7.3f}"
            )

    recent_count = min(100, len(episode_returns))
    print(
        f"Mean return over the last {recent_count} episodes: "
        f"{np.mean(episode_returns[-recent_count:]):.1f}"
    )
    return episode_returns


def plot_returns(episode_returns: list[float], output_path: Path) -> None:
    """Save episode returns and a moving average without opening a GUI."""
    episode_numbers = np.arange(1, len(episode_returns) + 1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(episode_numbers, episode_returns, alpha=0.35, label="Return")

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
        title="Online one-step Actor-Critic on CartPole-v1",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train online one-step Actor-Critic on CartPole-v1."
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("actor-critic-online-one-step-td.pdf"),
        help="Path of the training-return plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = OnlineActorCriticAgent(state_size, action_size)

    try:
        episode_returns = train(agent, env, args.episodes, args.seed)
    finally:
        env.close()

    plot_returns(episode_returns, args.output)
    print(f"Saved training plot to: {args.output}")


if __name__ == "__main__":
    main()
