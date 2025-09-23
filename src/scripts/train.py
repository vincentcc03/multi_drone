import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from src.envs.env import RLGamesEnv
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
        device="cuda"
    )
    return RLGamesEnvWrapper(env)

# -------------------------------
# 2️⃣ 注册 vecenv
# -------------------------------
vecenv.register(
    "RLGPUEnv", # vecenv 类型名
    lambda config_name, num_actors, **kwargs: create_env(num_envs=num_actors, **kwargs)
)

# -------------------------------
# 3️⃣ 注册环境到 rl-games 配置
# -------------------------------
env_configurations.register(
    "rlgpu",
    {
        "vecenv_type": "RLGPUEnv",
        "env_creator": lambda **kwargs: create_env(**kwargs)
    }
)

# -------------------------------
# 4️⃣ 修复后的配置参数
# -------------------------------
# 读取参数配置
with open("src/config/ppo_config.yaml", "r") as f:
    yaml_conf = yaml.safe_load(f)

if __name__ == "__main__":
    runner = Runner()
    runner.load(yaml_conf)  # 直接传 yaml_conf
    runner.run({"train": True, "play": False})