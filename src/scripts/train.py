import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from src.envs.env import RLGamesEnv# 替换为你实际文件路径
from src.envs.RLGamesEnvWrapper import RLGamesEnvWrapper

# -------------------------------
# 1️⃣ 环境构造函数
# -------------------------------
def create_env(config_name=None, num_envs=1, **kwargs):
    # 使用传入的并行数 num_envs（来自 params['config']['num_actors']）
    env = RLGamesEnv(
        config_name="env_config.yaml",
        traj_name="traj_config.yaml",
        num_envs=num_envs,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return RLGamesEnvWrapper(env)

# -------------------------------
# 2️⃣ 注册 vecenv
# -------------------------------
vecenv.register(
    "RLGPUEnv",                  # vecenv 类型名
    lambda config_name, num_actors, **kwargs: create_env(**kwargs)
)

# -------------------------------
# 3️⃣ 注册环境到 rl-games 配置
# -------------------------------
env_configurations.register(
    "custom_rlgpu",
    {
        "vecenv_type": "RLGPUEnv",
        "env_creator": lambda **kwargs: create_env(**kwargs)
    }
)

# -------------------------------
# 4️⃣ PPO/A2C 超参数（添加 algo）
# -------------------------------
params = {
    "algo": {
        "name": "a2c_continuous"
    },
    "config": {
        "name": "drone_transport_exp",
        "env_name": "custom_rlgpu",
        "num_actors": 16,
        "horizon_length": 128,
        "minibatch_size": 256,
        "learning_rate": 3e-4,
        "max_epochs": 1000,
        "entropy_beta": 0.01,
        "clip_ratio": 0.2,
        "gamma": 0.99,
        "reward_shaper": {},
        "env_config": {},
        "lr_schedule": "fixed",
        "e_clip": 0.2,
        "clip_value": 0.2,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "grad_norm": 0.5,
        "normalize_advantage": False,
        "use_gae": True,
        "gae_lambda": 0.95,
        "value_bootstrap": True,
        "entropy_regularization": 0.0,
        "normalize_input": False,
        "critic_coef": 0.5,
        # 新增 tau（默认 1.0）
        "tau": 1.0
    },
    "network": {
        "name": "actor_critic",
        "hidden_units": [256, 256],
        "activation": "relu"
    },
    "model": {
        "name": "continuous_a2c",
        "actor_units": [256, 256],
        "critic_units": [256, 256]
    }
}

# -------------------------------
# 5️⃣ 创建 Runner 并训练
# -------------------------------
# 推荐用 load + run 的调用方式，run 需要一个 args 字典（至少包含 'train'）
runner = Runner()
runner.load({'params': params})
runner.run({'train': True, 'play': False})
