import torch
import numpy as np
try:
    import gym
    from gym.spaces import Box, Discrete
except Exception:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete

from rl_games.common.ivecenv import IVecEnv
from src.envs.env import RLGamesEnv

class RLGamesEnvWrapper(IVecEnv):
    def __init__(self, env: RLGamesEnv):
        super().__init__()
        self.env = env 
        self.num_envs = env.num_envs
        self.device = env.device

    def reset(self):
        obs = self.env.reset()
        return obs

    def reset_done(self):
        return self.reset()

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, reward, done, info

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = self.env.get_env_info()
        obs_dim = info.get("obs_dim")
        act_dim = info.get("act_dim")
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        return {
            "observation_space": observation_space,
            "action_space": action_space,
            "agents": info.get("agents"),
        }

    def has_action_masks(self):
        return False

    def close(self):
        pass

    def seed(self, seed=None):
        # 可选实现
        np.random.seed(seed)
        torch.manual_seed(seed)
