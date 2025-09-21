import torch
from src.utils.read_yaml import load_config
from src.utils.computer import hat, Omega, quat_to_rot

class PayloadDynamicsSimSingle:
    def __init__(self):
        config = load_config("env_config.yaml")
        self.device = config.get("device")

        # 设置负载参数
        self.m_l = config.get("m_l", 1.0)
        J_l_value = config.get("J_l", 0.15)
        self.J_l = torch.eye(3, device=self.device) * J_l_value
        self.j_inv = torch.linalg.inv(self.J_l)

        r_g = config.get("r_g", [0.1, 0, 0])
        self.r_g = torch.tensor(r_g, dtype=torch.float32, device=self.device)

        self.g = config.get("g", 9.81)
        self.dt = config.get("dt", 0.01)
        self.e3 = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        # 初始化状态 (13,)
        self.state = torch.zeros(13, device=self.device)
        self.omega_dot = torch.zeros(3, device=self.device)  # 角加速度
        self.state[6] = 1.0  # 单位四元数

    def dynamics(self, state, input_force_torque):
        """
        state: (13,)
        input_force_torque: (6,)
        return: state_dot (13,)
        """
        p_l = state[0:3]       # (3,)
        v_l = state[3:6]       # (3,)
        q_l = state[6:10]      # (4,)
        omega_l = state[10:13] # (3,)

        F_l = input_force_torque[0:3]  # (3,)
        M_l = input_force_torque[3:6]  # (3,)

        R_l = quat_to_rot(q_l)  # (3,3)

        # ------------------ p_dot ------------------
        p_dot = R_l @ v_l  # (3,)

        # ------------------ v_dot ------------------
        omega_hat = hat(self.omega_dot)  # (3,3)
        omega_cross_rg = omega_hat @ self.r_g
        omega_cross_omega_cross_rg = omega_hat @ (v_l + omega_hat @ self.r_g)

        v_dot = -omega_cross_rg - omega_cross_omega_cross_rg + F_l/self.m_l \
                - R_l.T @ (self.g * self.e3)

        # ------------------ q_dot ------------------
        q_dot = 0.5 * (Omega(omega_l) @ q_l)

        # ------------------ omega_dot ------------------
        term1 = M_l - hat(self.r_g) @ (R_l.T @ (self.g * self.e3))
        term2 = -hat(omega_l) @ (self.J_l @ omega_l)
        term3 = -self.m_l * hat(self.r_g) @ (v_dot + omega_hat @ v_l)

        omega_dot = self.j_inv @ (term1 + term2 + term3)
        self.omega_dot = omega_dot  # 保存角加速度，供下一步使用
        return torch.cat([p_dot, v_dot, q_dot, omega_dot], dim=0)  # (13,)

    def rk4_step(self, input_force_torque):
        """
        input_force_torque: (6,)
        return: new_state (13,)
        """
        state = self.state
        dt = self.dt

        k1 = self.dynamics(state, input_force_torque)
        k2 = self.dynamics(state + 0.5 * dt * k1, input_force_torque)
        k3 = self.dynamics(state + 0.5 * dt * k2, input_force_torque)
        k4 = self.dynamics(state + dt * k3, input_force_torque)

        self.state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state[6:10] = self.state[6:10] / self.state[6:10].norm()  # 归一化四元数
        return self.state.clone()
