import torch
from src.utils.read_yaml import load_config
from src.utils.computer import hat

class CableDynamicsSimBatch:
    def __init__(self, config_path="env_config.yaml"):
        config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 1)

        # 从配置文件读取挂点
        di_list = config.get("di_list", [])
        self.n_cables = len(di_list)
        self.r_i = torch.tensor(di_list, dtype=torch.float32, device=self.device)  # (n,3)

        # 从配置文件读取绳子长度
        self.rope_length = config.get("rope_length", 1.0)

        # 状态: [d_i(3), omega_i(3), l_i(1), v_i(1)] × n
        self.state = torch.zeros(self.batch_size, self.n_cables, 8, device=self.device)
        self.state[:,:,6] = self.rope_length  # 初始化绳子长度为 rope_length
        self.state[:,:,0] = 1.0  # 初始方向默认 [1,0,0]

        # 仿真参数
        self.dt = config.get("dt", 0.01)
        self.t_max = config.get("t_max", 20.0)

    def dynamics(self, state, action):
        """
        state: (B,n,8)
        action: (B,n,4) -> [gamma(3), a]
        """
        B, n, _ = state.shape

        d = state[:,:,0:3]      # (B,n,3)
        omega = state[:,:,3:6]  # (B,n,3)
        l = state[:,:,6]        # (B,n)
        v = state[:,:,7]        # (B,n)

        gamma = action[:,:,0:3] # (B,n,3)
        a = action[:,:,3]       # (B,n)

        # d_dot = omega × d
        omega_hat = hat(omega.view(-1,3)).view(B,n,3,3)   # (B,n,3,3)
        d_dot = torch.einsum("bnij,bnj->bni", omega_hat, d)

        # omega_dot = gamma
        omega_dot = gamma

        # l_dot = v
        l_dot = v

        # v_dot = a
        v_dot = a

        return torch.cat([d_dot, omega_dot, l_dot.unsqueeze(-1), v_dot.unsqueeze(-1)], dim=2)  # (B,n,8)

    def rk4_step(self, action):
        state = self.state
        dt = self.dt
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * dt * k1, action)
        k3 = self.dynamics(state + 0.5 * dt * k2, action)
        k4 = self.dynamics(state + dt * k3, action)

        self.state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # 单位化方向向量 d_i
        self.state[:,:,0:3] = self.state[:,:,0:3] / self.state[:,:,0:3].norm(dim=2, keepdim=True)

        return self.state.clone()

    def compute_force_torque(self, R_l):
        """
        计算对负载的合力和合力矩
        R_l: (B,3,3) 负载旋转矩阵
        return: (B,6) -> [F_l(3), M_l(3)]
        """
        B = self.batch_size
        d = self.state[:,:,0:3]  # (B,n,3)
        l = self.state[:,:,6]    # (B,n)

        # 张力 (这里你可以定义控制律，这里简单用 t_i = k*(l0-l) )
        t = torch.clamp(l, 0, self.t_max)  # (B,n)

        # 每条绳子在负载坐标系的力 R^T t_i d_i
        f_i = torch.einsum("bij,bnj->bni", R_l.transpose(1,2), t.unsqueeze(-1)*d)  # (B,n,3)

        # 总力
        F_l = f_i.sum(dim=1)  # (B,3)

        # 力矩: r_i × f_i
        M_l = torch.cross(self.r_i.unsqueeze(0).expand(B,-1,-1), f_i, dim=2).sum(dim=1)  # (B,3)

        return torch.cat([F_l, M_l], dim=1)  # (B,6)
