"""A minimal demonstration of one REINFORCE policy-gradient update."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """Map a CartPole state to probabilities for the two possible actions."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # CartPole states contain: position, velocity, angle, angular velocity.
        hidden = F.relu(self.hidden(state))

        # Softmax converts the two output scores into probabilities that sum to 1.
        return F.softmax(self.output(hidden), dim=-1)


class Agent:
    """An agent whose stochastic policy is represented by PolicyNet."""

    def __init__(self, state_size: int, action_size: int):
        self.gamma = 0.98
        self.learning_rate = 0.0002
        self.policy = PolicyNet(state_size, action_size)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

    def action_probabilities(self, state) -> torch.Tensor:
        """Return π(a|s), the action probabilities for one environment state."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.policy(state_tensor).squeeze(0)

    def get_action(self, state) -> tuple[int, torch.Tensor]:
        """Sample an action from the current policy π(a|s)."""
        probabilities = self.action_probabilities(state)
        # 根据概率，采取行动
        action = Categorical(probabilities).sample().item()
        return action, probabilities


def main() -> None:
    env = gym.make("CartPole-v1")

    try:
        # Gymnasium reset() returns both the initial state and an info dictionary.
        state, _ = env.reset()

        agent = Agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
        )

        # Sample A_0 from the policy distribution π_theta(.|S_0).
        action, probabilities_before = agent.get_action(state)
        probability_before = probabilities_before[action].item()

        print("采取的动作：", "向左推" if action == 0 else "向右推")
        print("更新前该动作的概率：", probability_before)

        # This file demonstrates one update, so use an example sampled return G_0.
        # In a full episode, G_0 would be the discounted sum:
        # R_0 + gamma*R_1 + gamma^2*R_2 + ...
        sampled_return = 1.0

        # REINFORCE minimizes L = -G_0 log π_theta(A_0|S_0).
        # Minimizing this loss increases the probability of a positive-return action.
        loss = -sampled_return * torch.log(probabilities_before[action])

        # Clear old gradients, calculate new gradients, and update network parameters.
        agent.optimizer.zero_grad()
        # 求导
        loss.backward()
        # 跟新参数
        agent.optimizer.step()

        # Re-evaluate the SAME action to show how the update changed its probability.
        with torch.no_grad():
            probabilities_after = agent.action_probabilities(state)
        probability_after = probabilities_after[action].item()

        print("更新后该动作的概率：", probability_after)
        print("概率变化：", probability_after - probability_before)
    finally:
        env.close()


if __name__ == "__main__":
    main()
