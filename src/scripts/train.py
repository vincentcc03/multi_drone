import torch
import gym
from stable_baselines3 import PPO
from src.envs.env import Env   # 你的自定义环境
from src.utils.read_yaml import load_config


def train():
    # 读取配置
    config = load_config("env_config.yaml")

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建环境
    env = Env(batch_size=config["batch_size"])

    # 创建 PPO 模型
    model = PPO(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        device=device
    )

    # 训练
    model.learn(total_timesteps=config["total_timesteps"])

    # 保存模型
    model.save(config["train"]["save_path"])

    # 加载模型（测试时）
    #model = PPO.load(config["train"]["save_path"], env=env)

    # 测试
    obs = env.reset()
    for _ in range(config["env"]["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    train()
