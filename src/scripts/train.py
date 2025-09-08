import os
import yaml
from rl_games.common.env_configurations import register_env
from rl_games.torch_runner import Runner
from src.envs.env import Env

# 1. 环境注册
def create_env(**kwargs):
    return Env(**kwargs)

register_env('drone_env', lambda config_name, **kwargs: create_env(**kwargs))

# 2. 读取YAML配置（可直接用你已有的ppo_config.yaml）
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 3. 写入rl_games专用yaml配置（如已存在可跳过此步）
def write_rlgames_yaml(config, out_path):
    # 只保留params字段
    params = config.get('params', {})
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump({'params': params}, f, allow_unicode=True)

if __name__ == "__main__":
    # 配置文件路径
    user_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/ppo_config.yaml'))
    rlgames_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../drone_ppo.yaml'))

    # 读取并写入rl_games格式yaml
    config = load_yaml_config(user_config_path)
    write_rlgames_yaml(config, rlgames_config_path)

    # 4. 启动训练
    runner = Runner()
    runner.load(rlgames_config_path)
    runner.run()