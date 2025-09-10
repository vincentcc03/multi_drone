import os
import yaml
import torch
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
import gym

# 导入您的环境
from src.envs.env import Env  # 请替换为您实际的导入路径

class RLGamesEnvWrapper(gym.Env):
    """
    将您的环境包装成 rl_games 需要的格式
    """
    def __init__(self, **kwargs):
        self.env = Env()
        self.num_envs = self.env.envs
        
    def step(self, actions):
        # actions 形状: (num_envs * action_dim,) 需要重塑为 (num_envs, action_dim)
        actions = actions.reshape(self.num_envs, -1)
        obs, rewards, dones, infos = self.env.step(actions)
        
        # rl_games 期望的格式
        return obs, rewards, dones, infos
    
    def reset(self):
        obs = self.env.reset()
        return obs
    
    def get_number_of_agents(self):
        return self.num_envs
    
    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['agents'] = self.num_envs
        info['value_size'] = 1
        return info

def create_env(**kwargs):
    """环境创建函数"""
    return RLGamesEnvWrapper(**kwargs)

# 注册环境到 rl_games
vecenv.register('PayloadEnv', lambda **kwargs: create_env(**kwargs))
env_configurations.register('PayloadEnv', {
    'env_creator': create_env,
    'vecenv_type': 'gym',  # 改为 'gym'
    'env_kwargs': {},
})

def train():
    """训练函数"""
    print("初始化训练...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建训练器
    runner = Runner()
    with open('src/config/ppo_config.yaml', 'r', encoding='utf-8') as f:
        yaml_conf = yaml.safe_load(f)
    runner.load(yaml_conf)
    runner.reset()
    
    print("开始训练 Payload PPO...")
    
    # 开始训练
    runner.run({
        'train': True,
        'play': False,
        'checkpoint': '',
        'sigma': None
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Payload Environment PPO Training')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'resume'], 
                       default='train', help='运行模式')
    parser.add_argument('--checkpoint', type=str, default='', 
                       help='检查点路径 (用于test或resume模式)')
    parser.add_argument('--config', type=str, default='payload_ppo_config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 确保必要的目录存在
    os.makedirs('runs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    train()
