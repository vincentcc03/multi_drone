import torch
from src.utils.read_yaml import load_config
import numpy as np
import math
from src.utils.computer import hat

class CableDynamicsSimSingle:
    """
    单环境版本
    状态 per cable: [dir(3), omega(3), T, T_dot]
    输入 per cable: [gamma(3), a]
    """
    def __init__(self):
        cfg = load_config("env_config.yaml")
        self.device = cfg.get("device")

        self.n_cables = cfg.get("rope_num")
        rl = cfg.get("rl")  # 半径
        alpha = 2 * np.pi / self.n_cables
        r_i = []
        for i in range(self.n_cables):
            x = rl * math.cos(i * alpha)
            y = rl * math.sin(i * alpha)
            z = 0
            r_i.append([x, y, z])
        self.r_i = torch.tensor(r_i, dtype=torch.float32, device=self.device)  # (n, 3)

        # 初值 (n, 8): [dir(3), omega(3), T, T_dot]
        self.state = torch.zeros(self.n_cables, 8, device=self.device)
        dir_init = cfg.get("cable_initial_dirs", [[0, 0, 1]] * self.n_cables)  # (n,3)
        for i in range(self.n_cables):
            self.state[i, 6] = cfg.get("cable_initial_tensions", 2)  # 初始张力
            self.state[i, 0:3] = torch.tensor(dir_init[i], device=self.device)

        # 便捷成员变量
        self.dir = self.state[:, 0:3]    # (n, 3)
        self.omega = self.state[:, 3:6]  # (n, 3)
        self.T = self.state[:, 6]        # (n,)
        self.T_dot = self.state[:, 7]    # (n,)

        self.dt = cfg.get("dt", 0.01)

    # ---- 动力学 ----
    def dynamics(self, state, action):
        """
        state: (n, 8)
        action: (n, 4) [gamma(3), a]
        return: xdot (n, 8)
        """
        dir = state[:, 0:3]      # (n, 3)
        omega = state[:, 3:6]    # (n, 3)
        T = state[:, 6]          # (n,)
        T_dot = state[:, 7]      # (n,)

        gamma = action[:, 0:3]   # (n, 3)
        a = action[:, 3]         # (n,)

        # hat(omega) 对每根绳分别算
        dir_dot = torch.stack([hat(omega[i]) @ dir[i] for i in range(self.n_cables)], dim=0)  # (n, 3)
        omega_dot = gamma              # (n, 3)
        T_dot_new = T_dot              # (n,)
        T_ddot = a                     # (n,)

        xdot = torch.cat([
            dir_dot,                        # (n, 3)
            omega_dot,                      # (n, 3)
            T_dot_new.unsqueeze(-1),        # (n, 1)
            T_ddot.unsqueeze(-1)            # (n, 1)
        ], dim=1)  # (n, 8)

        return xdot

    def rk4_step(self, action):
        """
        action: (n, 4)
        return: state (n, 8)
        """
        dt = self.dt
        s = self.state

        k1 = self.dynamics(s, action)
        k2 = self.dynamics(s + 0.5 * dt * k1, action)
        k3 = self.dynamics(s + 0.5 * dt * k2, action)
        k4 = self.dynamics(s + dt * k3, action)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 保证 dir 为单位向量
        dir_norm = self.state[:, 0:3].norm(dim=1, keepdim=True) + 1e-8
        self.state[:, 0:3] = self.state[:, 0:3] / dir_norm

        # 更新便捷成员变量
        self.dir = self.state[:, 0:3]
        self.omega = self.state[:, 3:6]
        self.T = self.state[:, 6]
        self.T_dot = self.state[:, 7]

        return self.state.clone()

    # ---- 合力与合力矩（在负载坐标系表达）----
    def compute_force_torque(self, R_l):
        """
        R_l: (3,3) 负载姿态（世界->负载）
        return: (6,) = [F_l(3), M_l(3)]
        """
        # 世界系下绳方向
        d_world = self.dir  # (n, 3)

        # 各绳世界系力
        f_world = self.T.unsqueeze(-1) * d_world  # (n, 3)

        # 变换到负载坐标系
        f_l = (R_l.T @ f_world.T).T  # (n, 3)

        # 合力
        F_l = f_l.sum(dim=0)  # (3,)

        # 合力矩: r_i × f_i
        M_l = torch.cross(self.r_i, f_l, dim=1).sum(dim=0)  # (3,)

        return torch.cat([F_l, M_l], dim=0)  # (6,)
