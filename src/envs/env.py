import torch
import numpy as np
import os
import time
from datetime import datetime
from rl_games.common import env_configurations
from src.utils.load_traj import generate_complete_trajectory
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.utils.read_yaml import load_config
from src.utils.computer import quat_to_rot
from src.utils.plot_ppo import DroneVisualization

class RLGamesEnv:
    def __init__(self, config_name, traj_name, num_envs=1, device="cuda"):
        self.config = load_config(config_name)
        self.traj_config = load_config(traj_name)
        self.device = torch.device(self.config.get("device", device))
        self.num_envs = num_envs
        self.dt = self.config.get("dt")
        
        
        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch()
        self.cable = CableDynamicsSimBatch()

        # 参考轨迹
        self.ref_traj = generate_complete_trajectory()
        self.payload_init_state = torch.from_numpy(self.ref_traj['Ref_xl'][0]).float()
        print("Payload initial state:", self.payload_init_state)
        
        # 可视化
        self.visualizer = DroneVisualization()
        self.visualization_interval = 1  # 每10秒可视化一次
        self.last_visualization_time = 0
        self.ref_traj_vis = self.ref_traj['Ref_xl'][:, 0:3]  # 仅位置部分
        
        # 绳子初始状态
        self.rope_length = self.config.get("rope_length", 1.0)
        self.n_cables = self.cable.n_cables
        self.cable_init_state = self.cable.state.clone()

        # 无人机 / 障碍物参数
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos", []), dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r", 0.1)
        self.drone_radius = self.config.get("drone_radius", 0.125)
        self.r_payload = self.config.get("r_payload", 0.25)
        
        # 奖励权重
        self.pos_w = self.config.get("pos_weight", 1.0)
        self.vel_w = self.config.get("vel_weight", 0.1)
        self.quat_w = self.config.get("quat_weight", 0.1)
        self.omega_w = self.config.get("omega_weight", 0.01)
        
        # 终止条件参数
        self.collision_tolerance = self.config.get("collision_tolerance", 0.05)
        self.drone_high_distance = self.config.get("drone_high_distance", 0.5) #无人机间高度碰撞距离
        self.t_max = self.config.get("t_max", 100.0)

        # episode 计数
        self.current_point_index = 0
        self.step_counter = 0
        self.interval = self.traj_config["dt"]/self.config["dt"]

        # obs / act 空间
        self.obs_dim = 39
        self.act_dim = 4 * self.n_cables
        
        # 无人机位置
        self.drone_pos = torch.zeros(self.num_envs, self.n_cables, 3, device=self.device)

    # ============= rl-games 要求的方法 =============

    def reset(self):
        """批量 reset 所有环境"""
        B = self.num_envs
        self.payload.state = self.payload_init_state.unsqueeze(0).expand(B, 13).to(self.device)
        self.cable.state = self.cable_init_state.to(self.device)

        self.current_point_index = 0
        self.step_counter = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        action: torch.Tensor [B, act_dim]
        return: obs, reward, done, info
        """
        B = self.num_envs
        N = self.n_cables
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.view(B, N, 4)

        # 动力学仿真
        self.cable.rk4_step(action)
        
        q_l = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)
        self.payload.rk4_step(input_force_torque)

        # 计算无人机位置
        self.compute_drone_positions()

        # 奖励 + done
        reward, info = self._compute_reward(action)
        done = self._check_done()

        # 更新轨迹
        self.step_counter += 1
        
        if self.step_counter % self.interval == 0:
            self.current_point_index += 1
            
        # 可视化（每10秒一次）
        current_time = self.step_counter * self.dt
        if current_time - self.last_visualization_time >= self.visualization_interval:
            print(f"Visualizing at sim time: {current_time:.2f}s")
            self._visualize_current_state()
            self.last_visualization_time = current_time

        obs = self._get_obs()

        return obs, reward, done, info

    def get_number_of_agents(self):
        """rl-games 要求：返回 agent 数量"""
        return 1

    def get_env_info(self):
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "agents": 1
        }

    def _compute_reward(self, action):
        B = self.num_envs
        # 当前负载状态
        state = self.payload.state  # (B, 13)
        p_l = state[:, 0:3]         # (B, 3)
        v_l = state[:, 3:6]         # (B, 3)
        q_l = state[:, 6:10]        # (B, 4)
        omega_l = state[:, 10:13]   # (B, 3)

        # 参考状态
        next_idx = min(self.current_point_index + 1, len(self.ref_traj['Ref_xl']) - 1)
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][next_idx], 
                            dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, 13)
        p_l_ref = ref_xl[:, 0:3]
        v_l_ref = ref_xl[:, 3:6]
        q_l_ref = ref_xl[:, 6:10]
        omega_l_ref = ref_xl[:, 10:13]

        # 误差计算
        pos_error = torch.norm(p_l - p_l_ref, dim=1)
        vel_error = torch.norm(v_l - v_l_ref, dim=1)
        quat_error = 1.0 - torch.abs(torch.sum(q_l * q_l_ref, dim=1))
        omega_error = torch.norm(omega_l - omega_l_ref, dim=1)

        # 指数型奖励设计
        dt = self.config.get("dt")  # 仿真步长
        
        # 位置奖励：指数衰减
        pos_scale = self.config.get("pos_reward_scale")  # 距离缩放系数
        reward_pos = self.pos_w * torch.exp(-pos_error * pos_scale) * dt
        
        # 速度奖励：指数衰减
        vel_scale = self.config.get("vel_reward_scale")
        reward_vel = self.vel_w * torch.exp(-vel_error * vel_scale) * dt
        
        # 姿态奖励：指数衰减
        quat_scale = self.config.get("quat_reward_scale")  # 四元数误差通常较小
        reward_quat = self.quat_w * torch.exp(-quat_error * quat_scale) * dt
        
        # 角速度奖励：指数衰减
        omega_scale = self.config.get("omega_reward_scale")
        reward_omega = self.omega_w * torch.exp(-omega_error * omega_scale) * dt
        
        # 总奖励：正值，鼓励减小误差
        reward = reward_pos + reward_vel + reward_quat + reward_omega
        
        # 可选：添加额外奖励项
        # 1. 到达目标点的奖励
        if pos_error.min() < 0.1:  # 到达目标附近
            reward += 100 * dt
    
        # 2. 轨迹进展奖励
        progress_reward = self.config.get("progress_reward", 1)
        reward += progress_reward * dt
    
        # # 3. 碰撞惩罚（在_check_done中检测到碰撞时）
        # collision_penalty = self.config.get("collision_penalty", -10.0)
        # done = self._check_done()
        # reward = torch.where(done, collision_penalty, reward)
        
        # info
        info = []
        for i in range(B):
            info.append({
                'pos_error': pos_error[i].item(),
                'vel_error': vel_error[i].item(),
                'quat_error': quat_error[i].item(),
                'omega_error': omega_error[i].item(),
                'reward_pos': reward_pos[i].item(),
                'reward_vel': reward_vel[i].item(),
                'reward_quat': reward_quat[i].item(),
                'reward_omega': reward_omega[i].item(),
                'total_reward': reward[i].item(),
                'current_point_index': self.current_point_index,
                'target_position': p_l_ref[i],
                'current_position': p_l[i],
            })
        return reward, info

    def _check_done(self):
        B = self.num_envs
        # 每个环境一个 done 标志
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        # 1. 位置误差终止条件
        current_pos = self.payload.state[:, 0:3]   # (B, 3)
        next_idx = min(self.current_point_index + 1, len(self.ref_traj['Ref_xl']) - 1)
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][next_idx], 
                            dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, 13)
        ref_pos = ref_xl[:, 0:3]  # (B, 3)
        pos_error = torch.norm(current_pos - ref_pos, dim=1)  # (B,)
        done |= (pos_error > 10.0)

        # 2. 负载高度终止条件（负载高度小于等于0）
        payload_height = current_pos[:, 2]  # Z坐标 (B,)
        done |= (payload_height <= 0.0)

        # 3. 障碍物碰撞检测（假设 self.obstacle_pos 是 (N, 2)，平面上 N 个障碍物）
        if len(self.obstacle_pos) > 0:
            obs_pos = self.obstacle_pos.to(self.device)        # (N, 2)
            obs_pos = obs_pos.unsqueeze(0).expand(B, -1, -1)   # (B, N, 2)

            # 负载与障碍物碰撞检测
            payload_xy = current_pos[:, :2].unsqueeze(1)       # (B, 1, 2)
            dist_to_obs = torch.norm(payload_xy - obs_pos, dim=2)  # (B, N)
            min_dist, _ = torch.min(dist_to_obs, dim=1)            # (B,)
            done |= (min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance))

            # 无人机与障碍物碰撞检测
            drone_xy = self.drone_pos[:, :, :2].unsqueeze(2)   # (B, N_cables, 1, 2)
            obs_pos = obs_pos.unsqueeze(1)                     # (B, 1, N, 2)
            dist_to_obs = torch.norm(drone_xy - obs_pos, dim=3)  # (B, N_cables, N)
            min_dist, _ = torch.min(dist_to_obs, dim=2)          # (B, N_cables)
            done |= (min_dist < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any(dim=1)

            # 无人机之间的碰撞检测
            drone_xy = self.drone_pos[:, :, :2]  # (B, N_cables, 2)
            drone_z = self.drone_pos[:, :, 2]    # (B, N_cables)
            for i in range(self.n_cables):
                for j in range(i+1, self.n_cables):
                    # xy平面距离
                    dist_xy = torch.norm(drone_xy[:, i, :] - drone_xy[:, j, :], dim=1)  # (B,)
                    # z轴距离
                    dist_z = torch.abs(drone_z[:, i] - drone_z[:, j])  # (B,)
                    # xy距离和z距离都小于阈值才算碰撞
                    collision = (dist_xy < (2*self.drone_radius + self.collision_tolerance)) & \
                                (dist_z < self.drone_high_distance)
                    done |= collision

        # 4. 绳子拉力终止条件
        done |= (self.cable.T > self.t_max).any(dim=1)

        return done
    
    def _get_obs(self):
        B = self.num_envs

        # 1. rg_mag (B, 1)
        r_g_mag = self.payload.r_g.norm().item()
        r_g_mag = torch.full((B, 1), r_g_mag, device=self.device)  # (B, 1)

        # 2. 障碍物信息 (B, 6) - 假设最多2个障碍物
        obstacle_info = torch.zeros(B, 6, device=self.device)
        for i, pos in enumerate(self.obstacle_pos[:2]):  # 最多取前2个
            obstacle_info[:, i*3:(i+1)*3] = torch.cat([
                pos, torch.tensor([self.obstacle_r], device=self.device)
            ]).unsqueeze(0).expand(B, 3)

        # 3. ref_xl (B, 13) 和 ref_ul (B, 18)
        next_idx = min(self.current_point_index + 1, len(self.ref_traj['Ref_xl']) - 1)
        curr_idx = min(self.current_point_index, len(self.ref_traj['Ref_ul']) - 1)
        
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][next_idx], dtype=torch.float32, device=self.device)
        ref_ul = torch.tensor(self.ref_traj['Ref_ul'][curr_idx], dtype=torch.float32, device=self.device)
        ref_xl = ref_xl.unsqueeze(0).expand(B, -1)
        ref_ul = ref_ul.unsqueeze(0).expand(B, -1)
        
        # 4. drone_radius (B, 1)
        drone_radius = torch.full((B, 1), float(self.drone_radius), device=self.device)

        # 拼接
        obs = torch.cat([r_g_mag, obstacle_info, ref_xl, ref_ul, drone_radius], dim=1)  # (B, 1+6+13+18+1=39)
        return obs

    def render(self, mode="human"):
        pass
    
    def compute_drone_positions(self):
        """
        计算所有无人机在世界坐标系下的位置，返回 (B, N_cables, 3)
        """
        B = self.num_envs
        N = self.n_cables
        # 负载位置 (B, 3)
        p_load = self.payload.state[:, 0:3]  # (B, 3)
        # 负载姿态 (B, 4) -> (B, 3, 3)
        q_load = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_load)  # (B, 3, 3)
        # 绳子挂载点 (N, 3)
        r_i = self.cable.r_i  # (N, 3)
        # 绳子方向 (B, N, 3) 世界系下
        d_i = self.cable.dir  # (B, N, 3)
        # 绳子长度
        L = self.rope_length

        # 将绳子方向变换到负载坐标系下
        d_i_local = torch.einsum('bij,bnj->bni', R_l.transpose(1,2), d_i)  # (B, N, 3)
        # 负载系下无人机位置
        p_drone_load = r_i.unsqueeze(0) + d_i_local * L  # (B, N, 3)
        # 变换到世界系
        p_drone_world = torch.einsum('bij,bnj->bni', R_l, p_drone_load) + p_load.unsqueeze(1)  # (B, N, 3)
        self.drone_pos = p_drone_world
        
    def _visualize_current_state(self):
            """可视化当前状态（只可视化第一个环境）"""
            try:
                # 获取第一个环境的状态
                payload_pos = self.payload.state[0, 0:3].cpu().numpy()
                drone_positions = self.drone_pos[0].cpu().numpy()  # (N_cables, 3)
                
                # 障碍物位置（转换为numpy数组）
                obstacle_positions = self.obstacle_pos.cpu().numpy() if len(self.obstacle_pos) > 0 else np.array([])
                
                # 调用可视化函数
                self.visualizer.visualize_scene(
                    payload_pos=payload_pos,
                    drone_positions=drone_positions,
                    ref_trajectory=self.ref_traj_vis,
                    obstacle_positions=obstacle_positions,
                    obstacle_radius=self.obstacle_r,
                    payload_radius=self.r_payload,
                    drone_radius=self.drone_radius,
                    current_step=self.step_counter
                )
            except Exception as e:
                print(f"Visualization error: {e}")