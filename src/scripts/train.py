import torch
from src.utils.read_yaml import load_config
from src.envs.env import Env
from src.agent.agent1 import PPOAgent

def train():
    config = load_config("env_config.yaml")
    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    env = Env(batch_size=config["batch_size"], device=device)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, lr=config["learning_rate"])

    num_episodes = config["num_episodes"]
    max_steps = config["max_steps"]

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
            if done:
                break
        print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    train()