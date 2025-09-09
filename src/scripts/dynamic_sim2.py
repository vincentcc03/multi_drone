import os
import time
from src.utils.read_yaml import load_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.computer import quat_to_rot
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
import imageio


# 新建带时间戳的保存文件夹
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("results", "pictures", "dynamics", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 1. 初始化
rope = CableDynamicsSimBatch()
payload = PayloadDynamicsSimBatch()
cfg=load_config("env_config.yaml")
L=cfg["rope_length"]  # 绳子长度
steps = cfg["steps"]  # 仿真步数
B = rope.envs
n = rope.n_cables

frame_interval = steps // 20  # 每隔多少步保存一帧，共20帧
frame_paths = []
# 轨迹记录
payload_pos_traj = []
payload_rot_traj = []
rope_dir_traj = []

action = torch.zeros(B, n, 4, device=rope.device)  # [gamma(3), a]
action[:, :, 3] = 1  # 初始张力加速度
action[:, :, 0] = 1  # 初始角加速度
action[:, :, 2] = 0  # 初始角加速度

# 2. 仿真循环
for t in range(steps):
    rope.rk4_step(action)
    q_l = payload.state[:, 6:10]
    R_l = quat_to_rot(q_l)
    input_force_torque = rope.compute_force_torque(R_l)
    payload.rk4_step(input_force_torque)

    payload_pos_traj.append(payload.state[0, 0:3].cpu().numpy())
    payload_rot_traj.append(R_l[0].cpu().numpy())
    rope_dir_traj.append(rope.dir[0].cpu().numpy())

    # 每frame_interval步画一次（共20帧）
    if t % frame_interval == 0 or t == steps - 1:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        center = payload.state[0, 0:3].cpu().numpy()
        R = R_l[0].cpu().numpy()
        l = 1.0
        cube_pts = np.array([[0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[0.5,-0.5,-0.5],
                             [-0.5,0.5,0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5]])
        cube_pts = (R @ cube_pts.T).T + center
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        for e in edges:
            ax.plot(*zip(cube_pts[e[0]], cube_pts[e[1]]), color='b')

        for i in range(n):
            r_i = rope.r_i[i].cpu().numpy()
            attach = center + R @ r_i
            d = rope.dir[0, i].cpu().numpy()
            rope_end = attach + L*d
            ax.plot([attach[0], rope_end[0]], [attach[1], rope_end[1]], [attach[2], rope_end[2]], 'r-')
            ax.scatter(*rope_end, color='r', s=20)

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
        frame_path = os.path.join(save_dir, f'cube_rope_step_{t:03d}.png')
        plt.savefig(frame_path)
        plt.close()
        frame_paths.append(frame_path)

# 合成gif
gif_path = os.path.join(save_dir, "simulation.gif")
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for frame_path in frame_paths:
        image = imageio.imread(frame_path)
        writer.append_data(image)
print(f"GIF已保存到: {gif_path}")