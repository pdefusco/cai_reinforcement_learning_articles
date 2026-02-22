import gym

env = gym.make('CartPole-v1')
state = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    state = next_state

print(f"Episode finished with total reward: {total_reward}")
