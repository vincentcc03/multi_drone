import torch
import numpy as np
# 尝试先导入 gym，若不可用则回退到 gymnasium
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
        self.env = env 
        self.num_envs = env.num_envs
        self.device = env.device

    # ===== rl-games 需要的方法 =====
    def reset(self):
        obs = self.env.reset()
        return obs

    def reset_done(self):
        # 如果你环境内部没有部分 reset，就退化为全 reset
        return self.reset()

    def step(self, actions):
        """
        actions: torch.Tensor [num_envs, act_dim]
        return: obs, reward, done, info
        """
        obs, reward, done, info = self.env.step(actions)
        # 确保返回 numpy，rl-games 默认会再转 tensor
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
        return obs, reward, done, info

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = self.env.get_env_info()
        # 返回带 .shape 属性的空间对象，rl-games 期望这样
        obs_dim = info.get("obs_dim")
        act_dim = info.get("act_dim")
        # 使用 Float Box 作为默认观测/动作空间；如为离散动作请改为 Discrete(act_dim)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        return {
            "observation_space": observation_space,
            "action_space": action_space,
            "agents": info.get("agents"),
        }

    def has_action_masks(self):
        return False
