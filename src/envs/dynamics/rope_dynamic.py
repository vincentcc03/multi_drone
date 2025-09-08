import torch
from src.utils.read_yaml import load_config
from src.utils.computer import hat
class CableDynamicsSimBatch:
    """
    状态 per cable: [dir(3), omega(3), T, T_dot]
    输入 per cable: [gamma(3), a]
    """
    def __init__(self):
        cfg = load_config("env_config.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.envs = cfg.get("envs", 1)

        # 挂点在负载坐标系下的位置 r_i (n,3)
        di_list = cfg.get("di_list", [])
        self.n_cables = len(di_list)
        self.r_i = torch.tensor(di_list, dtype=torch.float32, device=self.device)

        # 初值
        dir_init = cfg.get("cable_initial_dirs", [[0, 0, 1]] * self.n_cables)  # (n,3)
        T_init = cfg.get("cable_initial_tensions", 1.0)

        # (B, n, 8): [dir(3), omega(3), T, T_dot]
        self.state = torch.zeros(self.envs, self.n_cables, 8, device=self.device)
        for i in range(self.n_cables):
            self.state[:, i, 0:3] = torch.tensor(dir_init[i], device=self.device)
            self.state[:, i, 6] = T_init if isinstance(T_init, (float, int)) else T_init[i]

        # 便捷成员变量
        self.dir = self.state[:, :, 0:3]      # (B, n, 3)
        self.omega = self.state[:, :, 3:6]    # (B, n, 3)
        self.T = self.state[:, :, 6]          # (B, n)
        self.T_dot = self.state[:, :, 7]      # (B, n)

        self.dt = cfg.get("dt", 0.01)

    # ---- 动力学 ----
    def dynamics(self, state, action):
        dir = state[:, :, 0:3]      # (B, n, 3)
        omega = state[:, :, 3:6]    # (B, n, 3)
        T = state[:, :, 6]          # (B, n)
        T_dot = state[:, :, 7]      # (B, n)

        gamma = action[:, :, 0:3]   # (B, n, 3)
        a = action[:, :, 3]         # (B, n)

        dir_dot = torch.matmul(hat(omega), dir.unsqueeze(-1)).squeeze(-1)  # (B, n, 3)
        omega_dot = gamma                         # (B, n, 3)
        T_dot_new = T_dot                         # (B, n)
        T_ddot = a                                # (B, n)

        xdot = torch.cat([
            dir_dot,              # (B, n, 3)
            omega_dot,            # (B, n, 3)
            T_dot_new.unsqueeze(-1),  # (B, n, 1)
            T_ddot.unsqueeze(-1)      # (B, n, 1)
        ], dim=2)  # (B, n, 8)
        return xdot

    def rk4_step(self, action):
        dt = self.dt
        s = self.state

        k1 = self.dynamics(s, action)
        k2 = self.dynamics(s + 0.5 * dt * k1, action)
        k3 = self.dynamics(s + 0.5 * dt * k2, action)
        k4 = self.dynamics(s + dt * k3, action)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 保证dir为单位向量
        dir_norm = self.state[:, :, 0:3].norm(dim=2, keepdim=True) + 1e-8
        self.state[:, :, 0:3] = self.state[:, :, 0:3] / dir_norm

        self.dir = self.state[:, :, 0:3]
        self.omega = self.state[:, :, 3:6]
        self.T = self.state[:, :, 6]
        self.T_dot = self.state[:, :, 7]

        return self.state.clone()

    # ---- 合力与合力矩（在负载坐标系表达）----
    def compute_force_torque(self, R_l):
        """
        R_l: (B,3,3) 负载姿态（世界->负载），即将世界向量变换到负载坐标的旋转矩阵
        返回 (B,6): [F_l(3), M_l(3)]  均在负载坐标系下
        """
        B = self.envs

        # 世界系下绳方向 d_i
        d_world = self.dir  # (B, n, 3)

        # 各绳世界系力
        f_world = self.T.unsqueeze(-1) * d_world  # (B, n, 3)

        # 变换到负载坐标系：f_l = R_l^T * f_world
        f_l = torch.einsum("bij,bnj->bni", R_l.transpose(1, 2), f_world)  # (B, n, 3)

        # 合力
        F_l = f_l.sum(dim=1)  # (B, 3)

        # 合力矩: r_i × f_i  (r_i 在负载坐标系中)
        r = self.r_i.unsqueeze(0).expand(B, -1, -1)  # (B, n, 3)
        M_l = torch.cross(r, f_l, dim=2).sum(dim=1)  # (B, 3)

        return torch.cat([F_l, M_l], dim=1)  # (B, 6)