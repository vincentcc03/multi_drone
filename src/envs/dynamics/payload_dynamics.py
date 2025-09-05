import torch
from src.utils.read_yaml import load_config
from src.utils.computer import hat, Omega, quat_to_rot

class PayloadDynamicsSimBatch:
    def __init__(self, config_path="env_config.yaml"):
        config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.envs = config.get("envs", 1)

        
        
        # 设置负载参数
        self.m_l = config.get("m_l", 1.0)
        J_l_value = config.get("J_l", 0.15)
        self.J_l = torch.eye(3, device=self.device) * J_l_value
        self.j_inv = torch.linalg.inv(self.J_l)
        
        r_g_list = config.get("r_g", [0.1, 0, 0])
        self.r_g = torch.tensor(r_g_list, dtype=torch.float32, device=self.device)
        
        self.g = config.get("g", 9.81)
        self.dt = config.get("dt", 0.01)
        self.e3 = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        # 初始化状态
        self.state = torch.zeros(self.envs, 13, device=self.device)
        self.state[:, 6] = 1.0  # 单位四元数te[:,6] = 1.0  # 单位四元数

    def dynamics(self, state, input_force_torque):
        B = self.envs
        device = self.device

        p_l = state[:, 0:3]       # (B,3)
        v_l = state[:, 3:6]       # (B,3)
        q_l = state[:, 6:10]      # (B,4)
        omega_l = state[:, 10:13] # (B,3)

        F_l = input_force_torque[:, 0:3]  # (B,3)
        M_l = input_force_torque[:, 3:6]  # (B,3)

        R_l = quat_to_rot(q_l)       # (B,3,3)

        # ------------------ p_dot ------------------
        p_dot = torch.bmm(R_l, v_l.unsqueeze(-1)).squeeze(-1)  # (B,3)

        # ------------------ v_dot ------------------
        r_g_exp = self.r_g.unsqueeze(0).expand(B, 3)  # (B,3)
        omega_hat = hat(omega_l)                 # (B,3,3)
        
        omega_cross_rg = torch.bmm(omega_hat, r_g_exp.unsqueeze(-1)).squeeze(-1)           # (B,3)
        omega_cross_omega_cross_rg = torch.bmm(omega_hat, (v_l + torch.bmm(omega_hat, r_g_exp.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)

        v_dot = -omega_cross_rg - omega_cross_omega_cross_rg + F_l/self.m_l \
                - torch.bmm(R_l.transpose(1,2), (self.g*self.e3).expand(B,3).unsqueeze(-1)).squeeze(-1)

        # ------------------ q_dot ------------------
        q_dot = 0.5 * torch.bmm(Omega(omega_l), q_l.unsqueeze(-1)).squeeze(-1)

        # ------------------ omega_dot ------------------
        # term1: M_l - hat(r_g) @ R.T @ g*e3
        term1 = M_l - torch.bmm(hat(r_g_exp), torch.bmm(R_l.transpose(1,2), (self.g*self.e3).expand(B,3).unsqueeze(-1))).squeeze(-1)
        
        # term2: - hat(omega) @ J @ omega
        term2 = -torch.bmm(omega_hat, torch.matmul(self.J_l, omega_l.T).T.unsqueeze(-1)).squeeze(-1)
        
        # term3: - m * hat(r_g) @ (v_dot + hat(omega)@v)
        term3 = -self.m_l * torch.bmm(hat(r_g_exp), (v_dot + torch.bmm(omega_hat, v_l.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        
        omega_dot = torch.matmul(self.j_inv, (term1 + term2 + term3).T).T  # (B,3)

        return torch.cat([p_dot, v_dot, q_dot, omega_dot], dim=1)  # (B,13)


    def rk4_step(self, input_force_torque):
        state = self.state
        dt = self.dt
        k1 = self.dynamics(state, input_force_torque)
        k2 = self.dynamics(state + 0.5 * dt * k1, input_force_torque)
        k3 = self.dynamics(state + 0.5 * dt * k2, input_force_torque)
        k4 = self.dynamics(state + dt * k3, input_force_torque)
        self.state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.state[:,6:10] = self.state[:,6:10] / self.state[:,6:10].norm(dim=1, keepdim=True)
        return self.state.clone()
