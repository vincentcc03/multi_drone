import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.computer import quat_to_rot
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch

# 新建带时间戳的保存文件夹
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("results", "pictures", "dynamics", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 1. 初始化
rope = CableDynamicsSimBatch(config_path="env_config.yaml")
payload = PayloadDynamicsSimBatch(config_path="env_config.yaml")

steps = 200
B = rope.envs
n = rope.n_cables

# 轨迹记录
payload_pos_traj = []
payload_rot_traj = []
rope_dir_traj = []

action = torch.zeros(B, n, 4, device=rope.device)  # [gamma(3), a]
action[:, :, 3] = 1  # 初始张力加速度

# 2. 仿真循环
for t in range(steps):
    rope.rk4_step(action)
    q_l = payload.state[:, 6:10]  # (B,4)
    R_l = quat_to_rot(q_l)        # (B,3,3)
    input_force_torque = rope.compute_force_torque(R_l)  # (B,6)
    payload.rk4_step(input_force_torque)

    payload_pos_traj.append(payload.state[0, 0:3].cpu().numpy())  # 只记录第一个env
    payload_rot_traj.append(R_l[0].cpu().numpy())
    rope_dir_traj.append(rope.dir[0].cpu().numpy())               # (n,3)

    # 每10步画一次
    if t % 10 == 0:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 画负载正方体
        center = payload.state[0, 0:3].cpu().numpy()  # 负载质心
        R = R_l[0].cpu().numpy()                     # 负载旋转
        l = 1.0  # 边长
        # 正方体8个顶点（以中心为原点，边长为1）
        cube_pts = np.array([[0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[0.5,-0.5,-0.5],
                             [-0.5,0.5,0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5]])
        # 旋转+平移
        cube_pts = (R @ cube_pts.T).T + center
        # 正方体12条边的顶点索引
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        for e in edges:
            ax.plot(*zip(cube_pts[e[0]], cube_pts[e[1]]), color='b')

        # 绳子
        for i in range(n):
            # 挂点在负载坐标系下
            r_i = rope.r_i[i].cpu().numpy()
            # 变换到世界系
            attach = center + R @ r_i
            # 绳方向（单位向量，世界系）
            d = rope.dir[0, i].cpu().numpy()
            # 绳子长度（可自定义，这里假设为1）
            rope_end = attach + d
            ax.plot([attach[0], rope_end[0]], [attach[1], rope_end[1]], [attach[2], rope_end[2]], 'r-')
            # 绳端点画个点
            ax.scatter(*rope_end, color='r', s=20)

        # 负载质心轨迹
        if t > 0:
            traj = np.stack(payload_pos_traj)
            ax.plot(traj[:,0], traj[:,1], traj[:,2], 'g--', label='Payload Traj')

        ax.set_xlim(center[0]-2, center[0]+2)
        ax.set_ylim(center[1]-2, center[1]+2)
        ax.set_zlim(center[2]-2, center[2]+2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Step {t}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cube_rope_step_{t:03d}.png'))
        plt.close()