import torch
from src.utils.plot import plot_trajectories, plot_trajectories_grid
from src.utils.read_yaml import load_config
# ---------------- 仿真类 ----------------
class PayloadDynamicsSimBatch:
    def __init__(self, batch_size, m_l, J_l=None, r_g=None, g=9.81, dt=0.01, initial_state=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.m_l = m_l
        self.J_l = J_l if J_l is not None else torch.eye(3, device=self.device)
        self.j_inv = torch.linalg.inv(self.J_l)
        self.r_g = r_g if r_g is not None else torch.zeros(3, device=self.device)
        self.g = g
        self.e3 = torch.tensor([0,0,1], dtype=torch.float32, device=self.device)
        self.dt = dt

        if initial_state is not None:
            self.state = initial_state.clone().to(self.device)  # (B, 13)
        else:
            self.state = torch.zeros(batch_size, 13, device=self.device)
            self.state[:,6] = 1.0  # 单位四元数

    @staticmethod
    def hat(v):
        B = v.shape[0]
        O = torch.zeros(B,3,3, dtype=v.dtype, device=v.device)
        O[:,0,1] = -v[:,2]; O[:,0,2] = v[:,1]
        O[:,1,0] = v[:,2];  O[:,1,2] = -v[:,0]
        O[:,2,0] = -v[:,1]; O[:,2,1] = v[:,0]
        return O

    @staticmethod
    def Omega(w):
        B = w.shape[0]
        O = torch.zeros(B,4,4, dtype=w.dtype, device=w.device)
        O[:,0,1:] = -w
        O[:,1:,0] = w
        O[:,1,2] = w[:,2]; O[:,1,3] = -w[:,1]
        O[:,2,1] = -w[:,2]; O[:,2,3] = w[:,0]
        O[:,3,1] = w[:,1];  O[:,3,2] = -w[:,0]
        return O

    @staticmethod
    def quat_to_rot(q):
        qw, qx, qy, qz = q[:,0], q[:,1], q[:,2], q[:,3]
        R = torch.zeros(q.shape[0],3,3, dtype=q.dtype, device=q.device)
        R[:,0,0] = 1 - 2*(qy**2+qz**2)
        R[:,0,1] = 2*(qx*qy-qz*qw)
        R[:,0,2] = 2*(qx*qz+qy*qw)
        R[:,1,0] = 2*(qx*qy+qz*qw)
        R[:,1,1] = 1 - 2*(qx**2+qz**2)
        R[:,1,2] = 2*(qy*qz-qx*qw)
        R[:,2,0] = 2*(qx*qz-qy*qw)
        R[:,2,1] = 2*(qy*qz+qx*qw)
        R[:,2,2] = 1 - 2*(qx**2+qy**2)
        return R

    def dynamics(self, state, input_force_torque):
        B = self.batch_size
        device = self.device

        p_l = state[:, 0:3]       # (B,3)
        v_l = state[:, 3:6]       # (B,3)
        q_l = state[:, 6:10]      # (B,4)
        omega_l = state[:, 10:13] # (B,3)

        F_l = input_force_torque[:, 0:3]  # (B,3)
        M_l = input_force_torque[:, 3:6]  # (B,3)

        R_l = self.quat_to_rot(q_l)       # (B,3,3)

        # ------------------ p_dot ------------------
        p_dot = torch.bmm(R_l, v_l.unsqueeze(-1)).squeeze(-1)  # (B,3)

        # ------------------ v_dot ------------------
        r_g_exp = self.r_g.unsqueeze(0).expand(B, 3)  # (B,3)
        omega_hat = self.hat(omega_l)                 # (B,3,3)
        
        omega_cross_rg = torch.bmm(omega_hat, r_g_exp.unsqueeze(-1)).squeeze(-1)           # (B,3)
        omega_cross_omega_cross_rg = torch.bmm(omega_hat, (v_l + torch.bmm(omega_hat, r_g_exp.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)

        v_dot = -omega_cross_rg - omega_cross_omega_cross_rg + F_l/self.m_l \
                - torch.bmm(R_l.transpose(1,2), (self.g*self.e3).expand(B,3).unsqueeze(-1)).squeeze(-1)

        # ------------------ q_dot ------------------
        q_dot = 0.5 * torch.bmm(self.Omega(omega_l), q_l.unsqueeze(-1)).squeeze(-1)

        # ------------------ omega_dot ------------------
        # term1: M_l - hat(r_g) @ R.T @ g*e3
        term1 = M_l - torch.bmm(self.hat(r_g_exp), torch.bmm(R_l.transpose(1,2), (self.g*self.e3).expand(B,3).unsqueeze(-1))).squeeze(-1)
        
        # term2: - hat(omega) @ J @ omega
        term2 = -torch.bmm(omega_hat, torch.matmul(self.J_l, omega_l.T).T.unsqueeze(-1)).squeeze(-1)
        
        # term3: - m * hat(r_g) @ (v_dot + hat(omega)@v)
        term3 = -self.m_l * torch.bmm(self.hat(r_g_exp), (v_dot + torch.bmm(omega_hat, v_l.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        
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

# ---------------- 批量仿真函数 ----------------
def simulate_batch(sim_class, batch_size, input_force_torque_seq, steps, **sim_kwargs):
    sims = sim_class(batch_size=batch_size, initial_state=torch.zeros(batch_size, 13, device=sim_kwargs.get("device", "cpu")), **sim_kwargs)
    sims.state[:,6] = 1.0  # 单位四元数
    
    trajs = []
    for i in range(steps):
        trajs.append(sims.state.clone())
        sims.rk4_step(input_force_torque_seq[i])
    return torch.stack(trajs)  # (steps, batch, 13)


if __name__ == "__main__":
    config = load_config("env_config.yaml")
    steps = config["steps"]
    batch_size = config["batch_size"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    input_force_torque_seq = torch.zeros(steps, batch_size, 6, device=device)
    for b in range(batch_size):
        input_force_torque_seq[:, b, 2] = 30 + b  # z方向力不同

    trajs = simulate_batch(
        PayloadDynamicsSimBatch,
        batch_size=batch_size,
        input_force_torque_seq=input_force_torque_seq,
        steps=steps,
        m_l=config["m_l"],
        J_l=config["J_l"]*torch.eye(3, device=device),
        r_g=torch.tensor(config["r_g"], device=device),
        g=config["g"],
        dt=config["dt"],
        device=device
    )
    plot_trajectories_grid(trajs)
    plot_trajectories(trajs)
    print(trajs.device)