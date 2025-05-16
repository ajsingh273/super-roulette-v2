from roulette_env import RouletteEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Initialize and check the custom environment
env = RouletteEnv()
check_env(env, warn=True)

# Train the RL agent
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000, exploration_fraction=0.2)
model.learn(total_timesteps=100_000)

# Save the model
model.save("roulette_dqn_agent")
