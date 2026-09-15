import gymnasium as gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

env = gym.make("CartPole-v1", render_mode="rgb_array")

state,  info = env.reset() # reset the state as S_0
done = False
# 将每一步的即时奖励R_t都保存在一个列表中
episode_rewards = []
# 回报（总的奖励）
G = 0.0

frames = []

# 折扣因子gamma：γ
gamma = 0.9

# 前向过程：玩一局游戏，采样一条轨迹τ(trajectory)
while not done:
    frames.append(env.render()) # 渲染每一步的图像并保存到数组中
    # 随机选择一个动作
    action = random.choice([0, 1])
    # 执行动作
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    # 将即时奖励R_t保存在数组中，前向计算
    episode_rewards.append(reward)

# 反向过程：逆序计算回报
# G_t = R_t + γG_{t+1}
# 最终计算出来的是G_0
for r in episode_rewards[::-1]:
    G = r + gamma * G

print("回报：", G)
env.close()


def show_animation(imgs):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    ax.text(10, 20, "", fontsize=12, color="black")

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


show_animation(frames)
