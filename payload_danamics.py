import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PayloadDynamicsSim:
    def __init__(self, m_l=1.0, J_l=None, r_g=None, g=9.81, dt=0.01, initial_state=None):
        self.m_l = m_l
        self.J_l = J_l if J_l is not None else torch.eye(3)
        self.j_inv = torch.linalg.inv(self.J_l)
        self.r_g = r_g if r_g is not None else torch.zeros(3)
        self.g = g
        self.e3 = torch.tensor([0,0,1])
        self.dt = dt
        self.v_dot = torch.zeros(3)  # 初始化 v_dot
        self.omega_dot = torch.zeros(3)  # 初始化 omega_dot
        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            self.state = torch.zeros(13)
            self.state[6] = 1.0  # 默认初始四元数为单位 (w,x,y,z)

    @staticmethod
    def hat(v):
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=v.dtype)

    @staticmethod
    def Omega(w):
        return torch.tensor([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ], dtype=w.dtype)

    @staticmethod
    def quat_to_rot(q):
        qw, qx, qy, qz = q
        R = torch.tensor([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ], dtype=q.dtype)
        return R

    def dynamics(self, state, input_force_torque):
        p_l = state[0:3]
        v_l = state[3:6]
        q_l = state[6:10]
        omega_l = state[10:13]
        e3 = self.e3
        F_l = input_force_torque[0:3]
        M_l = input_force_torque[3:6]
        R_l = self.quat_to_rot(q_l)

        p_dot = R_l @ v_l
        omega_cross_rg = self.hat(omega_l) @ self.r_g
        omega_cross_omega_cross_rg = self.hat(omega_l) @ (v_l + self.hat(omega_l) @ self.r_g)
        v_dot = -omega_cross_rg - omega_cross_omega_cross_rg + F_l/self.m_l - R_l.T @ (self.g * e3)
        q_dot = 0.5 * self.Omega(omega_l) @ q_l
        term1 = M_l - self.hat(self.r_g) @ R_l.T @ (self.g * e3)
        term2 = -self.hat(omega_l) @ (self.J_l @ omega_l)
        term3 = -self.m_l * self.hat(self.r_g) @ (v_dot + self.hat(omega_l) @ v_l)
        omega_dot = self.j_inv @ (term1 + term2 + term3)
        return torch.cat([p_dot, v_dot, q_dot, omega_dot])

    def rk4_step(self, input_force_torque):
        state = self.state
        dt = self.dt
        k1 = self.dynamics(state, input_force_torque)
        k2 = self.dynamics(state + 0.5 * dt * k1, input_force_torque)
        k3 = self.dynamics(state + 0.5 * dt * k2, input_force_torque)
        k4 = self.dynamics(state + dt * k3, input_force_torque)
        self.state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state[6:10] = self.state[6:10] / self.state[6:10].norm()
        return self.state.clone()
    
# 仿真和可视化函数（类外部）
def simulate(sim, input_force_torque_seq, steps):
    traj = []
    for i in range(steps):
        traj.append(sim.state.clone())
        input_ft = input_force_torque_seq[i]
        sim.rk4_step(input_ft)
    return torch.stack(traj)

def plot_trajectory(traj, save_path="trajectory_3d.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0].numpy(), traj[:,1].numpy(), traj[:,2].numpy(), label='xyz')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Load 3D Trajectory')
    ax.legend()
    plt.show()
    fig.savefig(save_path)

# 使用示例
if __name__ == "__main__":
    initial_state = torch.zeros(13)
    initial_state[6] = 1.0  # 初始四元数为单位 (w,x,y,z)
    sim = PayloadDynamicsSim(
        m_l=1.0,
        J_l=torch.eye(3),
        r_g=torch.tensor([0, 0, 0], dtype=torch.float32),
        g=9.81,
        dt=0.01,
        initial_state=initial_state
    )
    steps = 100
    input_force_torque_seq = torch.zeros(steps, 6)
    input_force_torque_seq[:,2] = 0  # 施加一个向上的力

    traj = simulate(sim, input_force_torque_seq, steps)
    plot_trajectory(traj)