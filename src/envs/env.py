import gymnasium as gym
import numpy as np
import torch
from src.utils.load_traj import generate_complete_trajectory
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimSingle
from src.envs.dynamics.rope_dynamic import CableDynamicsSimSingle
from src.utils.read_yaml import load_config
from src.utils.computer import quat_to_rot

class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.config = load_config("env_config.yaml")
        self.traj_config = load_config("traj_config.yaml")
        self.device = torch.device(self.config.get("device", "cpu"))

        # 轨迹点步进设置
        self.current_point_index = 0
        self.step_counter = 0
        self.interval = self.traj_config["dt"] / self.config["dt"]

        # 动力学仿真器（单环境）
        self.payload = PayloadDynamicsSimSingle()
        self.cable = CableDynamicsSimSingle()

        # 参考轨迹
        self.ref_traj = generate_complete_trajectory()  # 返回字典
        self.payload_init_state = torch.tensor(self.ref_traj['Ref_xl'][0], dtype=torch.float32, device=self.device)

        # 绳子初始状态
        self.rope_length = self.config.get("rope_length", 1.0)
        self.n_cables = self.cable.n_cables
        self.cable_init_state = self.cable.state.clone()

        # 障碍物信息
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos"), dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r")

        # 无人机状态
        self.drone_pos = torch.zeros(self.n_cables, 3, device=self.device)
        self.drone_radius = self.config.get("drone_radius", 0.125)
        self.r_payload = self.config.get("r_payload", 0.25)

        # 终止条件
        self.collision_tolerance = self.config.get("collision_tolerance", 0.1)
        self.drone_high_distance = self.config.get("drone_high_distance", 0.2)
        self.t_max = self.config.get("t_max", 50.0)

        # 奖励权重设置
        self.pos_w = self.config.get("pos_w", 1.0)
        self.vel_w = self.config.get("vel_w", 0.1)
        self.quat_w = self.config.get("quat_w", 0.5)
        self.omega_w = self.config.get("omega_w", 0.1)

        # 观测空间: rg模长(0) + 2*障碍物(1-6) + ref_x(7-19) + ref_ul(20-37) + drone_radius(38)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        # 动作空间: 角加速度3 + 拉力加速度1 = 4
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(4*self.n_cables,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 重置负载和绳子状态
        self.payload.state = self.payload_init_state.clone()
        self.cable.state = self.cable_init_state.clone()
        self.current_point_index = 0
        self.step_counter = 0
        # 可选：设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        obs = self._get_obs()
        info = {}  # 可以添加需要的环境信息
        return obs, info

    def step(self, action):
        # 1. 处理动作 (N, 4)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).view(self.n_cables, 4)

        # 2. 执行动作更新状态
        self.cable.rk4_step(action)
        q_l = self.payload.state[6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)
        self.payload.rk4_step(input_force_torque)

        # 计算无人机位置
        self.compute_drone_positions()
        # 3. 计算奖励
        reward, info_dict = self._compute_reward(action)

        # 4. 检查 done
        terminated = self._check_done()
        truncated = False  # 可根据需要设置截断条件（如最大步数），否则默认False

        # 5. 更新轨迹索引
        self.step_counter += 1
        if self.step_counter % self.interval == 0:
            self.current_point_index += 1

        # 6. 生成 obs（用新的 ref_{t+1}）
        obs = self._get_obs()

        info = info_dict
        reward = reward

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        # 当前负载状态
        state = self.payload.state  # (13,)
        p_l = state[0:3]         # (3,)
        v_l = state[3:6]         # (3,)
        q_l = state[6:10]        # (4,)
        omega_l = state[10:13]   # (3,)

        # 参考状态
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][self.current_point_index+1], dtype=torch.float32, device=self.device)
        p_l_ref = ref_xl[0:3]
        v_l_ref = ref_xl[3:6]
        q_l_ref = ref_xl[6:10]
        omega_l_ref = ref_xl[10:13]

        # 位置误差
        pos_error = torch.norm(p_l - p_l_ref)
        # 速度误差
        vel_error = torch.norm(v_l - v_l_ref)
        # 姿态误差（四元数距离，常用1-内积绝对值）
        quat_error = 1.0 - torch.abs(torch.sum(q_l * q_l_ref))
        # 角速度误差
        omega_error = torch.norm(omega_l - omega_l_ref)

        # reward 设计（可加权）
        reward = - (pos_error * self.pos_w + vel_error * self.vel_w + quat_error * self.quat_w + omega_error * self.omega_w)
        # info
        info = {
            'pos_error': pos_error.item(),
            'vel_error': vel_error.item(),
            'quat_error': quat_error.item(),
            'omega_error': omega_error.item(),
            'current_point_index': self.current_point_index,
            'target_position': p_l_ref.cpu().numpy(),
            'current_position': p_l.cpu().numpy(),
        }
        return reward.cpu().item(), info

    def _check_done(self):
        # 障碍物碰撞检测
        current_pos = self.payload.state[0:3]   # (3,)
        done = False
        if len(self.obstacle_pos) > 0:
            obs_pos = self.obstacle_pos.to(self.device)        # (N, 2)
            # 负载与障碍物碰撞检测
            payload_xy = current_pos[:2].unsqueeze(0)          # (1, 2)
            dist_to_obs = torch.norm(payload_xy - obs_pos, dim=1)  # (N,)
            min_dist = torch.min(dist_to_obs)
            if min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance):
                done = True
            # 无人机与障碍物碰撞检测
            drone_xy = self.drone_pos[:, :2]                   # (N_cables, 2)
            dist_to_obs = torch.norm(drone_xy.unsqueeze(1) - obs_pos.unsqueeze(0), dim=2)  # (N_cables, N)
            min_dist = torch.min(dist_to_obs)
            if (dist_to_obs < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any():
                done = True

            # 无人机之间的碰撞检测
            drone_z = self.drone_pos[:, 2]    # (N_cables,)
            for i in range(self.n_cables):
                for j in range(i+1, self.n_cables):
                    dist_xy = torch.norm(drone_xy[i] - drone_xy[j])
                    dist_z = torch.abs(drone_z[i] - drone_z[j])
                    if dist_xy < (2*self.drone_radius + self.collision_tolerance) and dist_z < self.drone_high_distance:
                        done = True

        # 绳子拉力终止条件
        if (self.cable.T > self.t_max).any():
            done = True

        return done

    def _get_obs(self):
        # 1. rg_mag (1,)
        r_g_mag = self.payload.r_g.norm().item()
        r_g_mag = torch.tensor([r_g_mag], device=self.device)  # (1,)

        # 2. 障碍物信息 (6,)
        obstacle_info = []
        for pos in self.obstacle_pos:
            obs_info = torch.cat([
                pos,
                torch.tensor([self.obstacle_r], device=self.device)
            ])
            obstacle_info.append(obs_info)
        obstacle_info = torch.cat(obstacle_info)  # (6,)

        # 3. ref_xl (13,) 和 ref_ul (18,)
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][self.current_point_index+1], dtype=torch.float32, device=self.device)
        ref_ul = torch.tensor(self.ref_traj['Ref_ul'][self.current_point_index], dtype=torch.float32, device=self.device)

        # 4. drone_radius (1,)
        drone_radius = torch.tensor([float(self.drone_radius)], device=self.device)

        # 拼接
        obs = torch.cat([r_g_mag, obstacle_info, ref_xl, ref_ul, drone_radius], dim=0)  # (39,)
        return obs.cpu().numpy()

    def render(self, mode="human"):
        pass

    def compute_drone_positions(self):
        """
        计算所有无人机在世界坐标系下的位置，返回 (N_cables, 3)
        """
        # 负载位置 (3,)
        p_load = self.payload.state[0:3]  # (3,)
        # 负载姿态 (4,) -> (3, 3)
        q_load = self.payload.state[6:10]
        R_l = quat_to_rot(q_load)  # (3, 3)
        # 绳子挂载点 (N, 3)
        r_i = self.cable.r_i  # (N, 3)
        # 绳子方向 (N, 3) 世界系下
        d_i = self.cable.dir  # (N, 3)
        # 绳子长度
        L = self.rope_length

        # 将绳子方向变换到负载坐标系下
        d_i_local = (R_l.transpose(0, 1) @ d_i.T).T  # (N, 3)
        # 负载系下无人机位置
        p_drone_load = r_i + d_i_local * L  # (N, 3)
        # 变换到世界系
        p_drone_world = (R_l @ p_drone_load.T).T + p_load.unsqueeze(0)  # (N, 3)
        self.drone_pos = p_drone_world