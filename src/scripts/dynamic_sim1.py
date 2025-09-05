import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.computer import quat_to_rot
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch

import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 初始化
rope = CableDynamicsSimBatch(config_path="env_config.yaml")
payload = PayloadDynamicsSimBatch(config_path="env_config.yaml")

steps = 200
B = rope.envs
n = rope.n_cables

# 轨迹记录
payload_pos_traj = []
rope_dir_traj = []
action = torch.zeros(B, n, 4, device=rope.device)  # [gamma(3), a]
action[:, :, 3] = 1  # 初始张力加速度
action[:, :, 0] = 1  # x方向角加速度
action[:, :, 1] = 1  # y方向角加速度
# 2. 仿真循环
for t in range(steps):
    # 你可以自定义action，这里用零输入做示例
    
    # action[:, :, 0:3] = ... # 可自定义角加速度
    # action[:, :, 3] = ...   # 可自定义张力加速度

    # 绳子动力学步进
    rope.rk4_step(action)

    # 负载姿态
    q_l = payload.state[:, 6:10]  # (B,4)
    R_l = quat_to_rot(q_l)        # (B,3,3)

    # 绳子输出合力/力矩，作为负载输入
    input_force_torque = rope.compute_force_torque(R_l)  # (B,6)

    # 负载动力学步进
    payload.rk4_step(input_force_torque)

    # 记录轨迹
    payload_pos_traj.append(payload.state[:, 0:3].cpu().numpy())  # (B,3)
    rope_dir_traj.append(rope.dir.cpu().numpy())                  # (B,n,3)

# 3. 可视化
payload_pos_traj = np.stack(payload_pos_traj, axis=0)  # (steps, B, 3)
rope_dir_traj = np.stack(rope_dir_traj, axis=0)        # (steps, B, n, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 画负载轨迹
for b in range(B):
    ax.plot(payload_pos_traj[:, b, 0], payload_pos_traj[:, b, 1], payload_pos_traj[:, b, 2], label=f'Payload {b}')

# 画每根绳子的末端轨迹
for i in range(n):
    rope_end = payload_pos_traj[:, 0, :] + rope.r_i[i].cpu().numpy() + rope_dir_traj[:, 0, i, :]
    ax.plot(rope_end[:, 0], rope_end[:, 1], rope_end[:, 2], '--', label=f'Rope {i}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# # 新建保存目录（带时间戳）
# timestamp = time.strftime("%Y%m%d_%H%M%S")
# save_dir = os.path.join("results", "pictures", timestamp)
# os.makedirs(save_dir, exist_ok=True)
# plt.savefig(os.path.join(save_dir, "sim_traj.png"))
plt.show()