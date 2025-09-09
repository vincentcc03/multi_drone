import gym
import numpy as np
import torch
from src.utils.load_traj import generate_complete_trajectory
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.utils.read_yaml import load_config
from src.utils.computer import quat_to_rot
class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.config = load_config("env_config.yaml")
        self.device = torch.device(self.config.get("device", "cpu"))
        self.envs = self.config.get("envs", 1)

        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch()
        self.cable = CableDynamicsSimBatch()
        
        # 参考轨迹
        self.ref_traj = generate_complete_trajectory() #返回字典
        # 从参考轨迹获取负载初始状态
        self.payload_init_state = self.ref_traj['Ref_xl'][0]  

        # 绳子长度
        self.rope_length = self.config.get("rope_length", 1.0)
        self.g = self.config.get("g", 9.81)
        self.n_cables = self.cable.n_cables
        self.cable_init_state = self.cable.state.clone()

        # 障碍物信息
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos"), dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r")

        # 无人机半径
        self.drone_radius = self.config.get("drone_radius", 0.125)
        
        
        # 终止条件
        self.max_tracking_error = self.config.get("max_tracking_error", 5.0)
        self.collision_tolerance = self.config.get("collision_tolerance", 0.1)
        self.trajectory_length = self.config.get("trajectory_length")

        # PPO相关参数
        self.total_timesteps = self.config.get("total_timesteps", 1000000)
        self.n_steps = self.config.get("n_steps", 2048)
        self.points_per_epoch = max(1, self.trajectory_length // (self.total_timesteps // self.n_steps))
        self.current_point_index = 0
        self.step_counter = 0

        # 观测空间: rg模长(0) + 2*障碍物(1-6) + ref_x(7-19) + ref_ul(20-37) + drone_radius(38)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)
        # 动作空间: 角加速度3 + 拉力加速度1 = 4
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(4*self.n_cables,), dtype=np.float32)

    def reset(self):
        B = self.envs
        # 重置负载和绳子状态
        self.payload.state = self.payload_init_state.unsqueeze(0).expand(B, 13).to(self.device)
        self.cable.state = torch.zeros(B, 1, 8, dtype=torch.float32, device=self.device)
        self.cable.state[:, 0, 2] = self.cable_init_tension  # 初始张力

        self.current_point_index = 0
        self.step_counter = 0
        return self._get_obs()

    def step(self, action):
        B = self.envs
        action = torch.tensor(action, dtype=torch.float32, device=self.device).view(B, 4)
        wd_dot = action[:, 0:3]
        tension_ddot = action[:, 3]

        cable_action = torch.zeros(B, 1, 4, device=self.device)
        cable_action[:, 0, 0:3] = wd_dot
        cable_action[:, 0, 3] = tension_ddot

        self.cable.rk4_step(cable_action)
        q_l = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)
        self.payload.rk4_step(input_force_torque)

        self.step_counter += 1
        if self.step_counter % self.points_per_epoch == 0:
            self.current_point_index = min(self.current_point_index + 1, self.trajectory_length - 1)

        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        done = self._check_done()
        # 并行返回
        if not isinstance(done, np.ndarray):
            done = np.array([done] * B)
        if not isinstance(reward, np.ndarray):
            reward = np.array([reward] * B)
        if isinstance(info, dict):
            info = [info for _ in range(B)]
        return obs, reward, done, info

    def _compute_reward(self, action):
        B = self.envs
        current_pos = self.payload.state[:, 0:3]
        target_pos = load_trajectory_point(self.current_point_index).unsqueeze(0).expand(B, 3)
        tracking_error = torch.norm(current_pos - target_pos, dim=1)
        reward = -tracking_error.cpu().numpy()
        info = []
        for i in range(B):
            info.append({
                'tracking_error': tracking_error[i].item(),
                'current_point_index': self.current_point_index,
                'target_position': target_pos[i].cpu().numpy(),
                'current_position': current_pos[i].cpu().numpy(),
            })
        return reward, info

    def _check_done(self):
        B = self.envs
        done = np.zeros(B, dtype=bool)
        if self.step_counter >= self.total_timesteps:
            done[:] = True
            return done
        current_pos = self.payload.state[:, 0:3]
        done |= (current_pos[:, 2] < self.min_height).cpu().numpy()
        target_pos = load_trajectory_point(self.current_point_index).unsqueeze(0).expand(B, 3)
        tracking_error = torch.norm(current_pos - target_pos, dim=1)
        done |= (tracking_error > self.max_tracking_error).cpu().numpy()
        for obs_pos in self.obstacle_pos:
            obs_pos_3d = torch.cat([obs_pos, current_pos[:, 2:3]], dim=0)
            dist_to_obs = torch.norm(current_pos - obs_pos_3d.unsqueeze(0), dim=1)
            done |= (dist_to_obs < self.obstacle_r + self.collision_tolerance).cpu().numpy()
        if self.current_point_index >= self.trajectory_length - 1:
            done[:] = True
        return done

    def _get_obs(self):
        B = self.envs
        r_g_mag = self.payload.r_g.norm(dim=1, keepdim=True)  # (B,1) 负载质心偏移向量的模长
        # 障碍物信息 (x, y, r) * 2
        obstacle_info = []
        for pos in self.obstacle_pos:
            obs_info = torch.cat([
                pos,
                torch.tensor([self.obstacle_r], device=self.device)
            ]).unsqueeze(0).expand(B, 3)
            obstacle_info.append(obs_info)
        obstacle_info = torch.cat(obstacle_info, dim=1)  # (B, 6)
        # 下一时刻轨迹点 (3)
        
        ref_pl = next_point['Ref_pl']  # (3, horizon+1)
        obs = torch.cat([r_g, obstacle_info, next_point], dim=1)
        return obs.cpu().numpy()

    def render(self, mode="human"):
        pass