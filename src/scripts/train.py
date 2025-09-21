import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.envs.env import Env  # 改成你环境类的路径

# ----------------------------
# 配置多环境数量
# ----------------------------
num_envs = 4  # 可以根据 CPU 核数和显存调整

# ----------------------------
# 创建多环境
# ----------------------------
def make_env():
    def _init():
        env = Env()
        return env
    return _init

# 使用 SubprocVecEnv（多进程加速）或 DummyVecEnv（单进程）
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
# vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])  # 如果你想用单进程也行

# ----------------------------
# 检查环境
# ----------------------------
check_env(vec_env.envs[0], warn=True)  # 检查其中一个环境即可

# ----------------------------
# 创建 PPO 模型
# ----------------------------
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048 // num_envs,  # 总步数被 num_envs 分摊
)

# ----------------------------
# 训练
# ----------------------------
model.learn(total_timesteps=10000)  # 可以先用小步数测试

# ----------------------------
# 保存模型
# ----------------------------
model.save("ppo_payload_cable_multi")

# ----------------------------
# 测试模型
# ----------------------------
obs = vec_env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)
    if dones.any():  # SubprocVecEnv/DummyVecEnv 返回数组
        obs[dones] = vec_env.reset()[dones]
