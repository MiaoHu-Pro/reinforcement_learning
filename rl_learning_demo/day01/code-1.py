
import gymnasium as gym

print(gym.__version__)
# 强化学习之父：Richard Sutton
env = gym.make("CartPole-v1")
state, info = env.reset() # 重置为S_0状态
# state includes:
# • Cart position
# • Cart velocity
# • Pole angle
# • Pole angular velocity
print("S_0: ", state)

action_space = env.action_space # 两个动作
print(action_space) # Discrete(2)


# 选择动作：向左推
action = 0
# 采取向左推的动作
next_state, reward, terminated, truncated, info = env.step(action)
"""
next_state:
    cart_position,
    cart_velocity,
    pole_angle,
    pole_angular_velocity,
    
reward: The reward received for this step. CartPole normally gives 1.0 for every successful step.

terminated: True when the episode ends because a terminal condition occurred—for example, the pole tilted too far or the cart moved outside the permitted area.

truncated: True when the episode is stopped by an external limit rather than failure. For CartPole-v1, this normally happens after the 500-step time limit.


    
"""
done = terminated or truncated
print("S_1: ", next_state)
print("R_0: ", reward)
print("是否结束：",  done )
