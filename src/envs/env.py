import gym
import numpy as np
import torch
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch

class Env(gym.Env):
    def __init__(self, batch_size=1, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.sim = PayloadDynamicsSimBatch(batch_size=batch_size, device=self.device)
        # 观测空间和动作空间可以根据你的需求调整
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

    def reset(self):
        # 重置仿真器状态
        self.sim.state = torch.zeros(self.batch_size, 13, device=self.device)
        self.sim.state[:,6] = 1.0  # 单位四元数
        obs = self.sim.state[0].cpu().numpy()
        return obs

    def step(self, action):
        # action: (6,) numpy array
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = self.sim.rk4_step(action_tensor)
        obs = next_state[0].cpu().numpy()
        # 你需要根据任务设计 reward 和 done
        reward = -np.linalg.norm(obs[0:3])  # 例如距离原点的负值
        done = False  # 根据任务终止条件设置
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        # 可选：实现可视化
        pass