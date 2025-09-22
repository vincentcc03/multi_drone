import torch
import numpy as np
from rl_games.common import env_configurations
from src.utils.load_traj import generate_complete_trajectory
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.utils.read_yaml import load_config
from src.utils.computer import quat_to_rot

class RLGamesEnv:
    def __init__(self, config_name, traj_name, num_envs=1, device="cpu"):
        self.config = load_config(config_name)
        self.traj_config = load_config(traj_name)
        self.device = torch.device(self.config.get("device", device))
        self.num_envs = num_envs
        
        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch()
        self.cable = CableDynamicsSimBatch()

        # 参考轨迹
        self.ref_traj = generate_complete_trajectory()
        self.payload_init_state = self.ref_traj['Ref_xl'][0]  

        # 绳子初始状态
        self.rope_length = self.config.get("rope_length", 1.0)
        self.n_cables = self.cable.n_cables
        self.cable_init_state = self.cable.state.clone()

        # 无人机 / 障碍物参数
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos"), dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r")
        self.drone_radius = self.config.get("drone_radius", 0.125)
        self.r_payload = self.config.get("r_payload", 0.25)

        # episode 计数
        self.current_point_index = 0
        self.step_counter = 0
        self.interval = self.traj_config["dt"]/self.config["dt"]

        # obs / act 空间
        self.obs_dim = 39
        self.act_dim = 4 * self.n_cables

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
        action = action.view(B, N, 4).to(self.device)

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
        B = self.envs
        # 当前负载状态
        state = self.payload.state  # (B, 13)
        p_l = state[:, 0:3]         # (B, 3)
        v_l = state[:, 3:6]         # (B, 3)
        q_l = state[:, 6:10]        # (B, 4)
        omega_l = state[:, 10:13]   # (B, 3)

        # 参考状态
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][self.current_point_index+1], 
                            dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, 13)
        p_l_ref = ref_xl[:, 0:3]
        v_l_ref = ref_xl[:, 3:6]
        q_l_ref = ref_xl[:, 6:10]
        omega_l_ref = ref_xl[:, 10:13]

        # 位置误差
        pos_error = torch.norm(p_l - p_l_ref, dim=1)
        # 速度误差
        vel_error = torch.norm(v_l - v_l_ref, dim=1)
        # 姿态误差（四元数距离，常用1-内积绝对值）
        quat_error = 1.0 - torch.abs(torch.sum(q_l * q_l_ref, dim=1))
        # 角速度误差
        omega_error = torch.norm(omega_l - omega_l_ref, dim=1)

        # reward 设计（可加权）
        reward = - (pos_error * self.pos_w + vel_error * self.vel_w + quat_error * self.quat_w + omega_error * self.omega_w)
        # info
        info = []
        for i in range(B):
            info.append({
                'pos_error': pos_error[i].item(),
                'vel_error': vel_error[i].item(),
                'quat_error': quat_error[i].item(),
                'omega_error': omega_error[i].item(),
                'current_point_index': self.current_point_index,
                'target_position': p_l_ref[i].cpu().numpy(),
                'current_position': p_l[i].cpu().numpy(),
            })
        return reward.cpu().numpy(), info

    def _check_done(self):
        B = self.envs
        # 每个环境一个 done 标志
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        # 3. 障碍物碰撞检测（假设 self.obstacle_pos 是 (N, 2)，平面上 N 个障碍物）
        current_pos = self.payload.state[:, 0:3]   # (B, 3
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
        # 绳子拉力终止条件
        done |= (self.cable.T > self.t_max).any(dim=1)

        return done.cpu().numpy()

    
# 观测空间: rg_mag(0) + 2*障碍物(1-6) + ref_x(7-19) + ref_ul(20-37) + drone_radius(38)
    def _get_obs(self):
        B = self.envs

        # 1. rg_mag (B, 1)
        r_g_mag = self.payload.r_g.norm().item()
        r_g_mag = torch.full((B, 1), r_g_mag, device=self.device)  # (B, 1)

        # 2. 障碍物信息 (B, 6)
        obstacle_info = []
        for pos in self.obstacle_pos:
            obs_info = torch.cat([
                pos,
                torch.tensor([self.obstacle_r], device=self.device)
            ]).unsqueeze(0).expand(B, 3)
            obstacle_info.append(obs_info)
        obstacle_info = torch.cat(obstacle_info, dim=1)  # (B, 6)

        # 3. ref_xl (B, 13) 和 ref_ul (B, 18)
        ref_xl = torch.tensor(self.ref_traj['Ref_xl'][self.current_point_index+1], dtype=torch.float32, device=self.device)
        ref_ul = torch.tensor(self.ref_traj['Ref_ul'][self.current_point_index], dtype=torch.float32, device=self.device)
        ref_xl = ref_xl.unsqueeze(0).expand(B, -1)
        ref_ul = ref_ul.unsqueeze(0).expand(B, -1)
        # 4. drone_radius (B, 1)
        drone_radius = torch.full((B, 1), float(self.drone_radius), device=self.device)

        # 拼接
        obs = torch.cat([r_g_mag, obstacle_info, ref_xl, ref_ul, drone_radius], dim=1)  # (B, 1+6+13+18+1=39)
        return obs.cpu().numpy()

    def render(self, mode="human"):
        pass
    
    def compute_drone_positions(self):
        """
        计算所有无人机在世界坐标系下的位置，返回 (B, N_cables, 3)
        """
        B = self.envs
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