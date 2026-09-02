"""Legacy PPO-Clip example for CartPole-v0.

This file uses the old Gym API and is intentionally kept as the legacy
implementation.  Detailed comments below connect each code block to the PPO
equations.  The modern Gymnasium rewrite is:

    code-3-ppo-clipped-objective-GAE.py

Why PPO is added after Actor-Critic with GAE
--------------------------------------------
Ordinary trajectory-batched Actor-Critic normally uses each collected
trajectory for one update. PPO safely reuses the same on-policy trajectory for
several update epochs. It records the probability assigned by the old policy
when the action was sampled and compares it with the changing new policy:

    r_t(theta)
      = pi_theta(A_t | S_t) / pi_old(A_t | S_t)
      = exp(log pi_theta(A_t | S_t) - log pi_old(A_t | S_t))

The clipped surrogate objective limits the incentive for this ratio to move
outside [1-epsilon, 1+epsilon]:

    L_clip(theta)
      = E_t[min(
            r_t(theta) * A_hat_t,
            clip(r_t(theta), 1-epsilon, 1+epsilon) * A_hat_t
        )]

PyTorch minimizes losses, so the actor minimizes ``-L_clip``. This file uses
epsilon=0.2, producing the interval [0.8, 1.2]. GAE supplies ``A_hat_t`` and a
return-like target for the critic.

Important legacy limitations
----------------------------
* ``gym`` and ``CartPole-v0`` are obsolete; use the modern rewrite normally.
* Old Gym combines true termination and time-limit truncation in ``done``.
* The complete trajectory is reused as one batch; there are no minibatches.
* The code is preserved for study, so comments explain its behavior without
  changing the original algorithm.
"""

import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def show_animation(imgs):
    """Convert previously collected RGB frames into MP4 and GIF animations.

    This helper is not called by the training loop below. In the legacy Gym
    API, frames would normally be obtained with ``env.render(mode="rgb_array")``.
    """
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    text = ax.text(10, 20, "", fontsize=12, color="black")

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # Step数表示
        frames.append(frame)

    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")
    ani.save("cartpole.gif", writer="pillow")

    plt.close(fig)
    return ani


def plot_loss(episode_list, return_list, filename):
    """Plot episode returns and save them to ``filename``."""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


class PolicyNet(nn.Module):
    """Actor network representing the categorical policy pi_theta(a | s).

    CartPole's state has four numbers and its action space has two actions. For
    a batch of T states, the output shape is [T, 2], and every row satisfies:

        pi_theta(a=0 | S_t) + pi_theta(a=1 | S_t) = 1.
    """

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):  # x contains one or more states S_t
        # Hidden representation h_t = ReLU(W_1 S_t + b_1).
        x = F.relu(self.l1(x))
        # Convert action scores into pi_theta(. | S_t) with softmax:
        #
        #   pi_theta(a | S_t) = exp(z_a) / sum_b exp(z_b).
        #
        # The modern script returns logits and lets Categorical handle this,
        # which is more numerically stable. This line preserves old behavior.
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    """Critic network approximating the state-value function V_omega(s).

    The value means the expected discounted return under the current policy:

        V^pi(s) = E_pi[R_t + gamma*R_(t+1) + ... | S_t=s].

    One scalar value is produced for every input state.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        # No softmax is used: a value estimate is a scalar, not a probability.
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    """Own the PPO actor, critic, data collection, GAE, and update logic."""

    def __init__(self):
        # Discount factor in returns and TD targets. A reward k steps in the
        # future receives weight gamma**k.
        self.gamma = 0.98

        # The actor and critic have separate parameters and optimizers. Their
        # losses are related, but actor gradients must not train the critic and
        # critic gradients must not train the actor.
        self.lr_pi = 0.001
        self.lr_v = 0.02
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        """Sample A_t ~ pi_old(. | S_t) and return all action probabilities.

        Sampling, instead of always taking argmax, supplies exploration. During
        collection ``self.pi`` is the old policy. It becomes the trainable new
        policy only when repeated PPO optimization begins.
        """
        # Add a batch dimension: state [4] -> [1, 4], then remove the output's
        # batch dimension: probabilities [1, 2] -> [2].
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # Categorical samples action 0 or 1 according to pi_old(. | S_t).
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs

    def collect_trajectory(self, env):
        """Collect one complete on-policy trajectory before any PPO update.

        One trajectory is the ordered transition sequence from initial state
        S_0 through the final state S_T:

            tau = (S_0,A_0,R_0,S_1, ..., S_(T-1),A_(T-1),R_(T-1),S_T).

        Why save the trajectory?
        * GAE at t depends on TD errors from t and later time steps.
        * PPO reuses these transitions for ten optimization epochs.
        * pi_old(A_t|S_t) must remain fixed while pi_theta changes.

        No ``backward()`` or optimizer ``step()`` occurs in this function, so
        all actions are generated by one unchanged old policy.
        """
        # Legacy reset API: old Gym returns only the state. Gymnasium returns
        # ``state, info`` instead.
        state = env.reset()
        states, next_states, actions, action_probs, rewards, dones = [], [], [], [], [], []
        done = False

        while not done:
            action, probs = self.get_action(state)
            # Legacy step API returns one ``done`` flag. Modern Gymnasium
            # returns separate ``terminated`` and ``truncated`` flags.
            next_state, reward, done, _ = env.step(action)

            # Save one transition plus the probability assigned by pi_old to
            # the action that was actually sampled.
            states.append(state)  # S_t
            next_states.append(next_state)  # S_(t+1)
            actions.append(action)  # A_t
            action_probs.append(probs[action])  # π_old(a_t|s_t)
            rewards.append(reward)  # R_t
            dones.append(done)  # done_t

            state = next_state

        # For a trajectory of T transitions, the aligned arrays are:
        #
        # states:       [S_0, S_1, ..., S_(T-1)]
        # next_states:  [S_1, S_2, ..., S_T]
        # actions:      [A_0, A_1, ..., A_(T-1)]
        # action_probs: [pi_old(A_0|S_0), ..., pi_old(A_(T-1)|S_(T-1))]
        # rewards:      [R_0, R_1, ..., R_(T-1)]
        # dones:        [False, False, ..., True]
        return states, next_states, actions, action_probs, rewards, dones

    def update(self, trajectory):
        """Reuse one trajectory for ten full-batch PPO update epochs.

        The update has four conceptual stages:

        1. Snapshot old critic values and old-policy log probabilities.
        2. Build one-step TD errors, GAE advantages, and critic targets.
        3. Re-evaluate the changing actor and form the clipped PPO loss.
        4. Run actor and critic backward/optimizer steps ten times.

        The old quantities and targets must remain fixed across all ten epochs.
        """
        states, next_states, actions, action_probs, rewards, dones = trajectory

        # Convert the complete trajectory into one tensor batch. Shapes:
        #
        # states, next_states: [T, 4]
        # actions:             [T, 1]
        # rewards, dones:      [T, 1]

        # # [𝑠0, 𝑠1, …, 𝑠𝑇 −1]
        states = torch.tensor(states)
        # # [𝑠1, 𝑠2, …, 𝑠𝑇 ]
        next_states = torch.tensor(next_states)
        # # [𝑎0, 𝑎1, …, 𝑎𝑇 −1]
        actions = torch.tensor(actions).view(-1, 1)
        # # [𝑅0, 𝑅1, …, 𝑅𝑇 −1]
        rewards = torch.tensor(rewards).view(-1, 1)
        # # [False1, False2, …, True𝑇 ]
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)

        # ------------------------------------------------------------------
        # Stage 1: construct one-step TD residuals for GAE
        # ------------------------------------------------------------------

        # Snapshot the old critic's current-state predictions:
        #
        #   [V_old(S_0), V_old(S_1), ..., V_old(S_(T-1))].
        #
        # detach() is essential: these old values later appear inside fixed
        # GAE critic targets and must not change during the ten PPO epochs.

        # # [𝑉 (𝑠0), 𝑉 (𝑠1), …, 𝑉 (𝑠𝑇 −1)]
        v = self.v(states).detach()

        # One-step TD target for each transition:
        #
        #   y_t^(1) = R_t + gamma*(1-done_t)*V_old(S_(t+1)).
        #
        # At the final ``done`` transition the bootstrap term becomes zero, so
        # y_(T-1) = R_(T-1). The old API cannot distinguish a true termination
        # from a time-limit truncation; the modern script handles that correctly.
        # # TD-target_𝑡 = 𝑅𝑡 + 𝛾𝑉 (𝑠𝑡+1)
        # # [TD-target0, TD-target1, …, TD-target𝑇 −1]
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        # One-step TD error (also called TD residual):
        #
        #   delta_t
        #     = y_t^(1) - V_old(S_t)
        #     = R_t + gamma*(1-done_t)*V_old(S_(t+1)) - V_old(S_t).
        #
        # ``td_delta`` contains [delta_0, ..., delta_(T-1)].

        # # 一步TD误差：𝛿𝑡 = 𝑅𝑡 + 𝛾𝑉 (𝑠𝑡+1) − 𝑉 (𝑠𝑡)
        # # [𝛿0, 𝛿1, …, 𝛿𝑇−1]
        td_delta = td_target - v

        # ------------------------------------------------------------------
        # Stage 2: combine future TD errors with GAE
        # ------------------------------------------------------------------

        # compute_gae() detaches td_delta and performs the backward recursion:
        #
        #   A_hat_t^GAE
        #     = delta_t + gamma*lambda*A_hat_(t+1)^GAE
        #     = delta_t
        #       + (gamma*lambda)*delta_(t+1)
        #       + (gamma*lambda)^2*delta_(t+2) + ...
        #
        # ``gae`` has shape [T, 1] and is a fixed actor learning signal.

        # # 计算每个时刻t的广义优势估计（GAE）
        gae = self.compute_gae(td_delta.cpu())

        # Freeze the probabilities recorded during collection. Reconstructing
        # a tensor here also separates them from the original actor graph.

        # # 冻结一份旧策略采取动作的对数概率log 𝜋𝜃old (𝑎𝑡|𝑠𝑡)
        # # [log 𝜋𝜃old (𝑎0|𝑠0)}, log 𝜋𝜃old (𝑎1|𝑠1)], …, log 𝜋𝜃old (𝑎𝑇 −1|𝑠𝑇 −1)}
        old_probs = torch.tensor(action_probs).view(-1, 1)

        # PPO stores log probabilities because subtraction in log space is more
        # numerically stable than directly dividing small probabilities:
        #
        #   log pi_old(A_t | S_t).
        old_log_probs = torch.log(old_probs).detach()

        # ------------------------------------------------------------------
        # Stages 3 and 4: reuse the trajectory for ten PPO epochs
        # ------------------------------------------------------------------

        # This repeated data reuse is the main distinction between PPO and the
        # preceding one-update Actor-Critic example. ``old_log_probs``, ``gae``,
        # ``v``, and the critic targets remain fixed while the networks change.
        for _ in range(10):
            # self.pi(states) has shape [T, 2]. gather(1, actions) selects the
            # current-policy probability of the action actually taken:
            #
            #   [pi_theta(A_0|S_0), ..., pi_theta(A_(T-1)|S_(T-1))].
            #
            # Taking log gives log pi_theta(A_t | S_t).
            #
            # We do not write the gradient
            #
            #   grad_theta log pi_theta(A_t | S_t)
            #
            # explicitly in Python. ``log_probs`` retains an autograd graph
            # connecting these log probabilities to the actor parameters
            # theta. Later, ``loss_pi.backward()`` follows that graph and
            # calculates the gradient automatically.
            log_probs = torch.log(self.pi(states).gather(1, actions))

            # PPO importance-sampling probability ratio:
            #
            #   r_t(theta)
            #     = pi_theta(A_t|S_t) / pi_old(A_t|S_t)
            #     = exp(log pi_theta(A_t|S_t) - log pi_old(A_t|S_t)).
            #
            # Before the first update, ratio is approximately 1. Later it shows
            # how far the updated policy moved on each collected action.
            #
            # Because old_log_probs is detached, only log_probs depends on
            # theta. The chain rule gives:
            #
            #   grad_theta r_t(theta)
            #     = r_t(theta)
            #       * grad_theta log pi_theta(A_t | S_t).
            ratio = torch.exp(log_probs - old_log_probs)

            # Unclipped surrogate objective:
            #
            #   surr1_t = r_t(theta) * A_hat_t^GAE.
            surr1 = ratio * gae

            # Clipped surrogate with epsilon=0.2:
            #
            #   surr2_t
            #     = clip(r_t(theta), 1-epsilon, 1+epsilon) * A_hat_t^GAE
            #     = clip(r_t(theta), 0.8, 1.2) * A_hat_t^GAE.
            #
            # Clipping removes the incentive for an already-large policy change
            # to keep increasing the surrogate objective.
            surr2 = torch.clamp(ratio, 0.8, 1.2) * gae

            # PPO maximizes the smaller (more conservative) surrogate:
            #
            #   J_clip(theta) = mean_t[min(surr1_t, surr2_t)].
            #
            # Optimizers perform gradient descent, so the code minimizes:
            #
            #   L_actor(theta) = -J_clip(theta).
            #
            # On the active, unclipped branch for transition t:
            #
            #   L_t(theta) = -r_t(theta) * A_hat_t.
            #
            # Its gradient contains the familiar policy-gradient term:
            #
            #   grad_theta L_t(theta)
            #     = -r_t(theta) * A_hat_t
            #       * grad_theta log pi_theta(A_t | S_t).
            #
            # PPO therefore still uses grad log pi. It is simply inside the
            # differentiable ratio and clipped objective instead of appearing
            # as a manually written gradient expression.
            loss_pi = torch.mean(-torch.min(surr1, surr2))

            # GAE also creates a return-like critic target:
            #
            #   V_hat_t^target = A_hat_t^GAE + V_old(S_t).
            #
            # ``gae`` and ``v`` are detached constants. Only the newly evaluated
            # self.v(states) on the left receives critic gradients:
            #
            #   L_critic(omega)
            #     = mean_t[(V_omega(S_t) - V_hat_t^target)^2].
            loss_v = F.mse_loss(self.v(states), gae + v)

            # Clear gradients left from the preceding PPO epoch. PyTorch
            # accumulates parameter gradients unless zero_grad() is called.
            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()

            # backward() computes gradients but does not change weights:
            #
            #   loss_v.backward()  -> gradients with respect to critic omega
            #   loss_pi.backward() -> gradients with respect to actor theta
            #
            # For the actor, autograd follows this dependency chain:
            #
            #   loss_pi
            #     -> min(surr1, surr2)
            #     -> ratio * GAE
            #     -> exp(log_probs - old_log_probs)
            #     -> log pi_theta(A_t | S_t)
            #     -> actor parameters theta.
            #
            # ``gae`` and ``old_log_probs`` are detached, so they are constants
            # in this differentiation. The gradient flows through the current
            # policy's ``log_probs`` only.
            loss_v.backward()
            loss_pi.backward()

            # step() is where the parameters actually change. After this, the
            # actor is a newer pi_theta, but old_log_probs remains pi_old.
            self.optimizer_pi.step()
            self.optimizer_v.step()

    def compute_gae(self, td_delta):
        """Calculate GAE for every time step by scanning TD errors backward.

        For lambda=0.95:

            A_hat_t^GAE
              = delta_t + gamma*lambda*A_hat_(t+1)^GAE.

        This is equivalent to the weighted sum:

            A_hat_t^GAE
              = delta_t
                + (gamma*lambda)*delta_(t+1)
                + (gamma*lambda)^2*delta_(t+2)
                + ...

        Backward calculation is efficient because the already accumulated
        ``last_gae`` contains all required future TD-error information.
        """
        # The advantages are training targets, not differentiable model
        # predictions. Detach and convert to NumPy to remove the autograd graph.
        td_delta = td_delta.detach().numpy()
        gae_list = []

        # Beyond the final collected transition there is no additional GAE
        # residual in this complete-episode implementation.
        last_gae = 0.0
        lmbda = 0.95

        # Iterate [delta_(T-1), ..., delta_1, delta_0]. For each t:
        #
        #   GAE_t = delta_t + gamma*lambda*GAE_(t+1).
        for delta in td_delta[::-1]:
            last_gae = delta + self.gamma * lmbda * last_gae
            gae_list.append(last_gae)

        # The recursion generated results in reverse temporal order. Restore:
        # [A_hat_0, A_hat_1, ..., A_hat_(T-1)].
        gae_list.reverse()
        return torch.tensor(gae_list)


# --------------------------------------------------------------------------
# Training workflow
# --------------------------------------------------------------------------

# Legacy environment construction and seeding. The modern equivalent is:
#
#   env = gymnasium.make("CartPole-v1")
#   state, info = env.reset(seed=42)
env = gym.make("CartPole-v0")
env.seed(42)
torch.manual_seed(42)
agent = Agent()
return_list = []
episode_list = []

for episode in range(500):
    # COLLECTION PHASE:
    # Generate exactly one new on-policy trajectory with fixed actor weights.
    trajectory = agent.collect_trajectory(env)

    # UPDATE PHASE:
    # update() calculates GAE once, then reuses this trajectory for ten actor
    # and critic optimizer steps. After the call returns, the trajectory is not
    # reused again; the next episode collects fresh data with the new policy.
    agent.update(trajectory)

    # trajectory[4] is the reward list [R_0, ..., R_(T-1)]. Its undiscounted
    # sum is the episode return shown in the learning curve.
    return_list.append(sum(trajectory[4]))
    episode_list.append(episode)
    if episode % 10 == 0:
        print(f"回合：{episode}, 总奖励：{sum(trajectory[4])}")

plot_loss(episode_list, return_list, "ppo-loss.pdf")
