import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime
from rl_games.common import env_configurations
from src.utils.load_traj import generate_complete_trajectory
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.utils.read_yaml import load_config
from src.utils.computer import quat_to_rot
from src.utils.plot_ppo import DroneVisualization
from src.utils.log import TrainingLogger
import random

# 控制print输出的开关
ENABLE_PRINT = True  # 设置为False即可关闭所有print

# 控制可视化的开关
ENABLE_VISUALIZATION = True  # 设置为False即可关闭所有可视化

# 控制俯视图的开关
ENABLE_TOP_VIEW = True  # 设置为False即可关闭俯视图绘制

# 重定义print函数
if not ENABLE_PRINT:
    def print(*args, **kwargs):
        pass


class RLGamesEnv:
    def __init__(self, config_name, traj_name, num_envs=1, device="cuda"):
        self.config = load_config(config_name)
        self.traj_config = load_config(traj_name)
        self.device = torch.device(self.config.get("device", device))
        self.num_envs = num_envs
        self.dt = self.config.get("dt")
        self.done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.logger = TrainingLogger()
        self.reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # 奖励相关
        self.last_dis = torch.zeros(self.num_envs, dtype=torch.double, device=self.device)
        self.last_pos_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_vel_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_quat_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_omega_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.angles = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.pos_error_all = torch.zeros(self.num_envs, 101, dtype=torch.float32, device=self.device)
        self.pos_error_1 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.achieve_num = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.payload_obstacle_min_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.min_drone_obstacle_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.min_drone_drone_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.traj_point_achieved = torch.zeros(self.num_envs, 101, dtype=torch.bool, device=self.device) #(B, 101)轨迹点达成情况
        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch()
        self.cable = CableDynamicsSimBatch()
        self.v_dot = self.payload.v_dot 
        self.omega_dot = self.payload.omega_dot
        self.r_g = self.payload.r_g
        self.r_i = self.cable.r_i
        # 奖励log
        self.reward_log = torch.zeros(8, dtype=torch.float32, device=self.device)
        self.end_pos_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.action_smoothness = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # 参考轨迹
        self.ref_traj = generate_complete_trajectory() #返回ref_xl, ref_ul等
        self.ref_xl = torch.as_tensor(self.ref_traj['Ref_xl'], dtype=torch.float32, device=self.device)
        self.ref_xl_end = self.ref_xl[-1].unsqueeze(0).expand(self.num_envs, -1)
        self.payload_init_state = self.ref_xl[0].clone()
        print("Payload initial state:", self.payload_init_state)
        # 各类奖励初始化
        self.pos_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.vel_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.quat_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.omega_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.traj_progress_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.drone_drone_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.drone_payload_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.action_smoothness_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.first_reach_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.traj_progress = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) #每一个环境的轨迹点最新位置

        # 可视化
        self.visualizer = DroneVisualization()
        self.ref_position = self.ref_traj['Ref_xl'][:, 0:3]  # 仅位置部分
        self.payload_traj_list = []
        self.payload_quat_traj_list = []
        for _ in range(self.num_envs):
            self.payload_traj_list.append([])
            self.payload_quat_traj_list.append([])
        self.env_done = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        
        # 绳子初始状态
        self.rope_length = self.config.get("rope_length", 1.0)
        self.n_cables = self.cable.n_cables
        self.cable_init_state = self.cable.state[0].clone()
        
        print("Cable initial state:", self.cable_init_state)

        # 无人机 / 障碍物参数
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos", []), dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r", 0.1)
        self.drone_radius = self.config.get("drone_radius", 0.125)
        self.r_payload = self.config.get("r_payload", 0.25)
        print(f"Obstacle positions: {self.obstacle_pos}, radius: {self.obstacle_r}")
        
        # 奖励权重
        self.pos_w = self.config.get("pos_weight", 1.0)
        self.vel_w = self.config.get("vel_weight", 0.1)
        self.quat_w = self.config.get("quat_weight", 0.1)
        self.omega_w = self.config.get("omega_weight", 0.01)
        
        # 终止条件参数
        self.collision_tolerance = self.config.get("collision_tolerance", 0.05)
        self.drone_high_distance = self.config.get("drone_high_distance", 0.5) #无人机间高度碰撞距离
        self.t_max = self.config.get("t_max", 100.0)
        self.pos_error_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # episode 计数
        self.current_point_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_counters = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.interval = int(round(self.traj_config["dt"] / self.config["dt"]))
        

        # obs / act 空间
        self.obs_dim = 3+6+13*3+1+13+4*8  # rg(3) + 障碍物(6) + 参考负载状态(13)*3 + 无人机半径(1) +真实状态（13）
        self.act_dim = 4 * self.n_cables
        self.action = torch.zeros(self.num_envs, self.n_cables, 4, dtype=torch.float32, device=self.device)
        self.action_smoothness = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # 无人机位置
        self.drone_pos = torch.zeros(self.num_envs, self.n_cables, 3, device=self.device)
        self.reset()
    # ============= rl-games 要求的方法 =============

    def reset(self): 
        """批量 reset 所有环境 - 向量化版本"""
        B = self.num_envs
        # 找出需要重置的环境（done为True的环境）
        reset_mask = self.done  # (B,) boolean tensor
        
        if reset_mask.any():
        
            # 向量化重置（性能更好）
            if reset_mask[0]:  # 如果第一个环境需要重置，清空其轨迹记录器
                self.draw_payload_traj_list()
                
            num_reset = reset_mask.sum().item()
            
            # ========== 核心修改：根据current_point_indices获取对应参考位置 ==========
            # 1. 生成随机的索引（1-10 或 90-100）
            choices = torch.randint(0, 4, size=(num_reset,), device=self.device) # 0,1,2,3
            front_start_point = torch.randint(0, 1, size=(num_reset,), device=self.device)  # 1-10
            back_start_point = torch.randint(60, 101, size=(num_reset,), device=self.device)  # 90-100
            random_nums = torch.where(choices == 5, back_start_point, front_start_point)
            self.current_point_indices[reset_mask] = random_nums
            print("Random indices for reset environments:", random_nums)
            
            # 2. 从参考轨迹中取出对应索引的位置（转为张量并适配设备）
            # 确保ref_position是张量，且索引不越界
            ref_traj_tensor = torch.tensor(self.ref_position, device=self.device, dtype=torch.float32)
            # 取出需要重置的环境对应的参考位置（random_nums是索引）
            payload_init_custom = ref_traj_tensor[random_nums]  # (num_reset, 3)
            
            # 3. 扩展初始状态：保留payload除位置外的其他维度（如速度、姿态等），仅替换位置
            # 先复制原始初始状态的结构
            payload_init_expanded = self.payload_init_state.unsqueeze(0).repeat(num_reset, 1).to(self.device)
            # 替换位置部分（前3维）为参考轨迹的对应位置
            payload_init_expanded[:, 0:3] = payload_init_custom
            # print("Payload initial states for reset environments:", payload_init_expanded)
            # 电缆初始状态保持不变
            cable_init_expanded = self.cable_init_state.unsqueeze(0).expand(num_reset, -1, -1).to(self.device)
            
            # 批量重置
            self.payload.state[reset_mask] = payload_init_expanded  # 用自定义初始位置赋值
            self.cable.state[reset_mask] = cable_init_expanded
            self.reward[reset_mask] = 0.0
            self.pos_error_count[reset_mask] = 0
            self.traj_progress[reset_mask] = 0
            self.env_done[reset_mask] = 0
            
            # 修正：last_dis计算改为当前初始位置到最后参考点的距离
            # self.last_dis[reset_mask] = torch.norm(self.payload.state[reset_mask, 0:3] - torch.tensor(self.ref_traj['Ref_xl'][-1][0:3], device=self.device), dim=1)
            
            self.last_pos_error[reset_mask] = 0.0
            self.last_vel_error[reset_mask] = 0.0
            self.last_quat_error[reset_mask] = 0.0
            self.last_omega_error[reset_mask] = 0.0
            self.angles[reset_mask] = 0.0
            self.achieve_num[reset_mask] = 0
            self.traj_point_achieved[reset_mask] = False
            
            # 重置计数器（确保interval是整数，避免类型不匹配）
            self.step_counters[reset_mask] = self.current_point_indices[reset_mask] * self.interval
            
            self.done[reset_mask] = False
        
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
        # 每五步更新一次action
        update_mask = (self.step_counters % 5 == 0)
        action = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), action, self.action)
        self.action_smoothness = -torch.norm(action - self.action, dim=(1,2))
        self.action = action
        print("Action:", self.action[0])

        # 动力学仿真
        self.cable.rk4_step(action)
        
        q_l = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)
        print("Input force and torque to payload:", input_force_torque[0])
        self.payload.rk4_step(input_force_torque)

        print("Payload state:", self.payload.state[0])
        # 检查负载和绳子状态是否有NaN，返回是第几个状态量有NaN
        nan_payload = torch.isnan(self.payload.state).any()
        nan_payload_indices = torch.isnan(self.payload.state).nonzero(as_tuple=True)
        nan_cable = torch.isnan(self.cable.state).any()
        if nan_payload :
            print("NaN detected in payload!")
            print("NaN indices in payload state:", nan_payload_indices)
        if nan_cable:
            print("NaN detected in cable!")
            
        # 计算无人机位置
        self.compute_drone_positions()
        
        

        # 奖励 + done
        self.done , info = self._check_done()
        reward = self._compute_reward(action)
        self.reward += reward
        # print("total_rewards:", self.reward)
        # max_reward = self.reward.max().item()
        # print(f"max reward: {max_reward}")
        self.reset()
        
        
        
        # 更新环境计数器
        self.step_counters += 1
        # 打印所有环境当前步数
        print(f"Current step: {self.step_counters}")
        
        # 检查哪些环境需要更新轨迹点
        update_mask = (self.step_counters % int(self.interval) == 0)
        self.current_point_indices[update_mask] += 1
        print(f"Current point indices: {self.current_point_indices}")
        
        # 可视化
        current_time = self.step_counters[0] * self.dt
        # print(f"Current sim time: {current_time:.2f}s")
        if ENABLE_VISUALIZATION and current_time*100 % 4 == 0:
            print(f"Visualizing at sim time: {current_time:.2f}s")
            # self._visualize_current_state()
            self._record_payload_trajectories()

        obs = self._get_obs()
        self.logger.log_step(
                self.achieve_num[0].item(), 
                self.reward[0].item(),
                self.reward_log[0].item(), 
                self.reward_log[1].item(), 
                self.reward_log[2].item(), 
                self.reward_log[3].item(), 
                self.reward_log[4].item(), 
                self.reward_log[5].item(),
                self.reward_log[6].item(),
                self.reward_log[7].item()
            )
        
        return obs, reward, self.done, info

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

        # === 2. 批量计算索引 ===
        lead_point1 = torch.clamp(self.current_point_indices, max=100)  # (B,)
        lead_point2 = torch.clamp(self.current_point_indices+3, max=100)  # (B,)
        lead_point3 = torch.clamp(self.current_point_indices+5, max=100)
        # next_id = torch.clamp(self.current_point_indices+1, max=100)  # (B,)
        # print(f"Next indices: {lead_point}")
        # === 3. 批量 gather 对应轨迹点 ===
        ref_xl_batch1 = self.ref_xl[lead_point1]  # (B, 13)
        ref_xl_batch2 = self.ref_xl[lead_point2]  # (B, 13)
        ref_xl_batch3 = self.ref_xl[lead_point3]  # (B, 13)
        # p_l_ref2 = self.ref_xl[next_id][:, 0:3]
        p_l_ref1 = ref_xl_batch1[:, 0:3]
        v_l_ref1 = ref_xl_batch1[:, 3:6]
        q_l_ref1 = ref_xl_batch1[:, 6:10]
        omega_l_ref1 = ref_xl_batch1[:, 10:13]
        p_l_ref2 = ref_xl_batch2[:, 0:3]
        v_l_ref2 = ref_xl_batch2[:, 3:6]
        q_l_ref2 = ref_xl_batch2[:, 6:10]
        omega_l_ref2 = ref_xl_batch2[:, 10:13]
        p_l_ref3 = ref_xl_batch3[:, 0:3]
        v_l_ref3 = ref_xl_batch3[:, 3:6]
        q_l_ref3 = ref_xl_batch3[:, 6:10]
        omega_l_ref3 = ref_xl_batch3[:, 10:13]    
        
        # 计算每个环境当前位置与轨迹上所有点的距离 (B, T)
        T = self.ref_xl.shape[0] #101
        # self.ref_xl[:,0:3] -> (T,3), p_l.unsqueeze(1) -> (B,1,3) -> broadcast -> (B,T,3)
        self.pos_error_all = torch.norm(p_l.unsqueeze(1) - self.ref_xl[:, 0:3].unsqueeze(0), dim=2)  # (B, T)
        self.pos_error_1 = torch.norm(p_l - p_l_ref1, dim=1)  # (B,)
        # 如果到达了某个轨迹点，则标记为已达成，如果有新到达的点就获得奖励
        reached_mask = self.pos_error_all < 0.1  # (B, T) bool tensor
        # 要是轨迹的新点才给奖励，如果第三十个到过了，就算29是新到达也不给奖励
        idx = torch.arange(T, device=self.traj_point_achieved.device)
        pre_max_idx = (self.traj_point_achieved * idx).max(dim=1).values
        self.traj_point_achieved[reached_mask] = True
        max_id = (self.traj_point_achieved * idx).max(dim=1).values
        achieve_num = self.traj_point_achieved.sum(dim=1)  # (B,)
        mask = max_id > pre_max_idx
        new_point_reward = mask.float() * 1.5
        # 如果achieve_num没达到17就不给奖励
        new_point_reward = torch.where(achieve_num < 20, torch.zeros_like(new_point_reward), new_point_reward)
        self.achieve_num = achieve_num
        reward = new_point_reward * 0
        # self.reward_log[7] = new_point_reward[0]
        
        

        
        # 误差计算
        pos_error = torch.norm(p_l - p_l_ref1, dim=1)+ torch.norm(p_l - p_l_ref2, dim=1)*0.5 + torch.norm(p_l - p_l_ref3, dim=1) *0.25
        vel_error = torch.norm(v_l - v_l_ref1, dim=1) + torch.norm(v_l - v_l_ref2, dim=1)*0.5 + torch.norm(v_l - v_l_ref3, dim=1) *0.25
        # 如果速度误差很小，给予5的奖励，如果achieve_num没达到17就不给奖励
        v_error_reward = torch.where(vel_error < 0.1, torch.full_like(vel_error, 5.0), torch.zeros_like(vel_error))
        v_error_reward = torch.where(achieve_num < 20, torch.zeros_like(v_error_reward), v_error_reward)
        # reward += v_error_reward
        # 四元数误差计算 计算三个相加 没做
        cos_half_theta = torch.abs(torch.sum(q_l * q_l_ref1, dim=1))
        cos_half_theta = torch.clamp(cos_half_theta, -1.0, 1.0)
        delta_theta = 2.0 * torch.acos(cos_half_theta)  # 弧度误差 ∈ [0, π]
        quat_error = delta_theta
        
        omega_error = torch.norm(omega_l - omega_l_ref1, dim=1) + torch.norm(omega_l - omega_l_ref2, dim=1)*0.5 + torch.norm(omega_l - omega_l_ref3, dim=1) *0.25
        # print ("p_l_ref:", p_l_ref)
        # print(f"Position errors: {pos_error}")
        # print(f"Velocity errors: {vel_error}")
        # print(f"Quaternion errors: {quat_error}")
        # print(f"Omega errors: {omega_error}")
        
        # 10开始，每隔十个点检测一次有没有跟踪上，如果跟踪上了就给较大奖励
        traj_reward = torch.where(
            (self.step_counters % 80 == 0) & (self.step_counters>100) & (self.pos_error_1 < 0.1),
            torch.full_like(self.pos_error_1, 10.0),
            torch.zeros_like(self.pos_error_1)
        )

        reward += traj_reward

        self.reward_log[7] = traj_reward[0]
        # pos_error = pos_error_all.min(dim=1).values
        # 位置奖励：指数衰减

        reward_pos = self.pos_w * torch.exp(-2*pos_error)
        # 当current_point_indices小于25时，奖励只有0.1倍
        reward_pos = torch.where(self.achieve_num < 20, reward_pos * 0.1, reward_pos)
        
        # 速度奖励：指数衰减
        reward_vel = self.vel_w * torch.exp(-2*vel_error)
        reward_vel = torch.where(self.achieve_num < 20, reward_vel * 0.1, reward_vel)
        # 姿态奖励：指数衰减
        reward_quat = self.quat_w * torch.exp(-3*quat_error)

        # 角速度奖励：指数衰减
        reward_omega = self.omega_w * torch.exp(-3 * omega_error)

        self.reward_log[0] = reward_pos[0]
        self.reward_log[1] = reward_vel[0]
        self.reward_log[2] = reward_quat[0]
        self.reward_log[3] = reward_omega[0]
        reward += reward_pos + reward_vel + reward_quat + reward_omega
        # self.last_pos_error = pos_error.clone()
        # self.last_vel_error = vel_error.clone()
        # self.last_quat_error = quat_error.clone()
        # self.last_omega_error = omega_error.clone()
        # 无人机负载间距惩罚：当距离小于0.1m才有惩罚，给予-5惩罚
        distance_threshold = 0.1
        drone_obstacle = self.min_drone_obstacle_dist.min(dim=1).values - self.drone_radius - self.obstacle_r  # (B,)
        drone_obstacle_reward = torch.where(
            drone_obstacle < distance_threshold,
            torch.full_like(drone_obstacle, -5.0),
            torch.zeros_like(drone_obstacle)
        )
        drone_obstacle_reward *= 0
        reward += drone_obstacle_reward
        # 无人机间距惩罚：当距离小于0.1m才有惩罚，-5
        drone_drone_dist = self.min_drone_drone_dist  # (B,)
        drone_drone_dist_reward = torch.where(
            drone_drone_dist < distance_threshold,
            torch.full_like(drone_drone_dist, -5.0),
            torch.zeros_like(drone_drone_dist)
        )
        # drone_drone_dist_reward *= 0
        reward += drone_drone_dist_reward
        
        # 角度惩罚
        angle_reward = -self.angles*0.4
        # reward += angle_reward
        # 高度惩罚
        height_reward = -abs(0.5 - p_l[:, 2])
        # reward += height_reward * 0.1
        self.reward_log[4] = drone_obstacle_reward[0]
        self.reward_log[5] = drone_drone_dist_reward[0]
        
        
        
        # 进度奖励
        # pos_errornow = torch.norm(p_l - p_l_ref2, dim=1)
        # reach = pos_errornow < 0.05
        # last_progress = self.traj_progress.clone()
        # self.traj_progress[reach] = self.current_point_indices[reach]
        # first_reach = (self.traj_progress != last_progress) & (self.current_point_indices > 11)
        # # 第一次到达了这个
        # reward += first_reach.float() * 50

        # 终止惩罚，到达轨迹点个数越多越好
        # penalty_mask = (self.done & (self.env_done == 1) | (self.env_done == 6)|(self.env_done == 7))  # 因碰撞或拉力过大终止
        # penalty_mask = self.done 
        # traj_progress_reward = penalty_mask.float() * (self.traj_point_achieved.sum(dim=1)/100 - 1) * 300
        # reward += traj_progress_reward
        # self.reward_log[7] = traj_progress_reward[0]
        # # 高度下降惩罚
        # reward += (0.5-p_l[:, 2]) * -1 
        # 终点距离奖励
        end_pos = torch.tensor(self.ref_traj['Ref_xl'][-1][0:3], device=self.device)
        dis_to_end = torch.norm(p_l - end_pos, dim=1)
        self.end_pos_reward =  -dis_to_end
        # reward += self.end_pos_reward
        
        # dis_to_end = torch.norm(p_l - torch.tensor(self.ref_traj['Ref_xl'][-1][0:3], device=self.device), dim=1)
        # self.end_pos_reward = (0.99*self.last_dis - dis_to_end) * 10 
        # self.last_dis = dis_to_end.clone()
        # reward = self.end_pos_reward
        # reward += 0.01
        # 可选：添加额外奖励项
        # 1. 到达目标点的奖励
        # if pos_error.min() < 0.1:  # 到达目标附近
        #     reward += 100 * dt
    
        # 动作平滑奖励
        self.action_smoothness *= 0.1
        self.action_smoothness *= 0
        # progress_reward = self.config.get("progress_reward", 1)
        # reward += progress_reward * dt
        reward += self.action_smoothness
        self.reward_log[6] = self.action_smoothness[0]
        # print(f"action_smoothness: {self.action_smoothness[0].item():.4f}, end_pos_reward: {self.end_pos_reward[0].item():.4f}")
        # print(f"Reward components : {reward_pos.mean().item():.6f}, {reward_vel.mean().item():.6f}, {reward_quat.mean().item():.6f}, {reward_omega.mean().item():.6f}")
        
        return reward

    def _check_done(self):
        B = self.num_envs
        current_pos = self.payload.state[:, 0:3]   # (B, 3)
        # 每个环境一个 done 标志
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        # NaN 检查：负载或绳子状态出现 NaN 则直接终止该环境
        # payload_nan = torch.isnan(self.payload.state).any(dim=1)
        # rope_state_flat = self.cable.state.reshape(B, -1)
        # rope_nan = torch.isnan(rope_state_flat).any(dim=1)
        # done = payload_nan | rope_nan
        self.env_done[done] += 1
        # 负载速度约束 大于3m/s
        load_velocity = self.payload.state[:, 3:6]  # (B, 3)
        done |= (torch.norm(load_velocity, dim=1) > 3.0)
        self.env_done[done] += 1
        # 无人机位置约束
        # pay_load_height = current_pos[:, 2]  # (B, N_cables)
        # done |= (self.drone_pos[:, :, 2] < pay_load_height[:, None]).any(dim=1)
        # self.env_done[done] += 1
        # # 1. 位置误差终止条件 - 为每个环境使用独立的轨迹索引
        
        # for i in range(B):
        #     next_idx = min(self.current_point_indices[i].item() + 1, len(self.ref_traj['Ref_xl']) - 1)
        #     ref_xl = torch.tensor(self.ref_traj['Ref_xl'][next_idx], 
        #                         dtype=torch.float32, device=self.device)
        #     ref_pos = ref_xl[0:3]  # (3,)
        #     pos_error = torch.norm(current_pos[i] - ref_pos)  # scalar
        #     if pos_error > self.config.get("pos_error_threshold", 0.25):  # 位置误差过大
        #         self.pos_error_count[i] += 1

        #     if self.pos_error_count[i] >= self.config.get("pos_error_count", 5):  # 连续5步误差过大
        #         done[i] = True
        #         print(f"Env {i} done due to position error. Count: {self.pos_error_count[i].item()}")
        # print("pos error counts:", self.pos_error_count)
        # if done.all():
        #     print("Position errors:")
        # 2. 负载高度终止条件（负载高度小于等于0）
        payload_height = current_pos[:, 2]  # Z坐标 (B,)

        # done |= (payload_height <= 0.4)
        # self.env_done[done] += 1
        # if (payload_height <= 0.0).any():
        #     # print("payload heights:", payload_height)
        # if done.all():
            # print("Payload heights:")
            # print("payload_state:", self.payload.state)
        # 3. 障碍物碰撞检测（假设 self.obstacle_pos 是 (N, 2)，平面上 N 个障碍物）
        if len(self.obstacle_pos) > 0:
            obs_pos = self.obstacle_pos.to(self.device)        # (N, 2)
            obs_pos = obs_pos.unsqueeze(0).expand(B, -1, -1)   # (B, N, 2)

            # 负载与障碍物碰撞检测
            payload_xy = current_pos[:, :2].unsqueeze(1)       # (B, 1, 2)
            dist_to_obs = torch.norm(payload_xy - obs_pos, dim=2)  # (B, N)
            self.payload_obstacle_min_dist, _ = torch.min(dist_to_obs, dim=1)            # (B,)
            done |= (self.payload_obstacle_min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance))
            # if (min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance)).any():
            #     print("Min distances from payload to obstacles:", min_dist)
            self.env_done[done] += 1
            # 无人机与障碍物碰撞检测
            drone_xy = self.drone_pos[:, :, :2].unsqueeze(2)   # (B, N_cables, 1, 2)
            obs_pos = obs_pos.unsqueeze(1)                     # (B, 1, N, 2)
            dist_to_obs = torch.norm(drone_xy - obs_pos, dim=3)  # (B, N_cables, N)
            self.min_drone_obstacle_dist, _ = torch.min(dist_to_obs, dim=2)          # (B, N_cables)
            # self.min_drone_obstacle_dist = min_dist.min(dim=1).values - self.drone_radius - self.r_payload  # (B,)
            # done |= (self.min_drone_obstacle_dist < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any(dim=1)
            # if (min_dist < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any():
            #     print("Min distances from drones to obstacles:", min_dist)
            self.env_done[done] += 1
            # if done.all():
                # print("Min distances from drones to obstacles:")
                
            # ------------------ 无人机-无人机碰撞检测 ------------------
        drone_pos_xyz = self.drone_pos  # (B, N, 3)
        B, N, _ = drone_pos_xyz.shape
        # 扩展维度以计算所有无人机之间的两两距离
        pos_i = drone_pos_xyz.unsqueeze(2)  # (B, N, 1, 3)
        pos_j = drone_pos_xyz.unsqueeze(1)  # (B, 1, N, 3)
        # 计算无人机间的欧几里得距离
        dist = torch.norm(pos_i - pos_j, dim=-1)  # (B, N, N)
        # 生成 mask，忽略自身距离（对角线）
        mask = ~torch.eye(N, dtype=torch.bool, device=dist.device).unsqueeze(0)  # (1, N, N)
        dist_masked = dist.masked_fill(~mask, float('inf'))  # 自身距离设为无穷大
        # 判断是否碰撞：任意两架无人机的距离小于安全阈值则判定碰撞
        collision_threshold = 2.5 * self.drone_radius + self.collision_tolerance
        self.min_drone_drone_dist = dist_masked.min(dim=2).values.min(dim=1).values - self.drone_radius * 2

        collision = dist_masked < collision_threshold  # (B, N, N)
        # 若任意两架无人机碰撞，则该环境 done = True
        done |= collision.any(dim=(1, 2))
        
        self.env_done[done] += 1

                    
        # 4. 绳子拉力终止条件
        done |= ((self.cable.T > self.t_max) | (self.cable.T < 0.0)).any(dim=1)
        # if (self.cable.T > self.t_max).any():
        #     print("Max cable tensions:", self.cable.T.max(dim=1).values)
        self.env_done[done] += 1
        # 绳子夹角终止条件
        # curr_dir = self.cable.dir  # (B, N, 3)
        # init_dir = self.cable_init_state[:, 0:3]  # (N, 3)
        # init_dir = init_dir.unsqueeze(0).expand(B, -1, -1).to(self.device)  # (B, N, 3)
        # cos_angle = torch.clamp((curr_dir * init_dir).sum(dim=2), -1.0, 1.0)  # (B, N)
        # angles = torch.acos(cos_angle)  # (B, N), 单位：弧度
        # max_allowed = torch.pi / 2  # 90 度
        # # done |= (angles > max_allowed).any(dim=1)
        # self.angles = angles.max(dim=1).values
        # self.env_done[done] += 1
        # if done.all():
        #     print("Max cable tensions:") 
        # 时间终止条件
        done |= (self.current_point_indices >= len(self.ref_traj['Ref_xl']) - 1)
        self.env_done[done] += 1
        # 离终点距离最大偏移终止条件
        # dis_to_end = torch.norm(current_pos - torch.tensor(self.ref_traj['Ref_xl'][-1][0:3], device=self.device), dim=1)
        # done |= (dis_to_end > 5)
        # 偏离轨迹过远终止条件，
        pos_error_all_min = self.pos_error_all.min(dim=1).values
        # done |= (self.pos_error_1 > 1.0)
        done |= (pos_error_all_min > 0.2)
        self.env_done[done] += 1
            
        info = []
        return done, info

    def _get_obs(self):
        B = self.num_envs

        r_g = self.r_g.unsqueeze(0).expand(B, -1).to(self.device)
        # 2️⃣ 障碍物信息 (B, 6) - 最多2个障碍物
        pad_obs = torch.zeros(2, 2, device=self.device)
        pad_obs[:2] = self.obstacle_pos[:2]
        obs_r = torch.full((2, 1), self.obstacle_r, device=self.device)
        obstacle_info = torch.cat([pad_obs, obs_r], dim=1).reshape(1, -1).expand(B, -1)

        # 3️⃣ ref_xl 和 ref_ul 批量索引
        T_xl = self.ref_xl.shape[0]
        next_idx1 = torch.clamp(self.current_point_indices + 1, max=T_xl - 1)
        next_idx2 = torch.clamp(self.current_point_indices + 3, max=T_xl - 1)
        next_idx3 = torch.clamp(self.current_point_indices + 5, max=T_xl - 1)
        # 取三个点，如果id超过100则重复取后面的的点
        ref_xl_batch = torch.cat([
            self.ref_xl[next_idx1].unsqueeze(1),
            self.ref_xl[next_idx2].unsqueeze(1),
            self.ref_xl[next_idx3].unsqueeze(1)
        ], dim=1).reshape(B, -1)


        # 4️⃣ drone_radius (B, 1)
        drone_radius = torch.full((B, 1), float(self.drone_radius), device=self.device)
        rope_state = self.cable.state.reshape(B, -1)
        # 5️⃣ 拼接观测向量
        obs = torch.cat([r_g, self.payload.state, obstacle_info, ref_xl_batch, drone_radius, rope_state], dim=1)
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
                    ref_trajectory=self.ref_position,
                    obstacle_positions=obstacle_positions,
                    obstacle_radius=self.obstacle_r,
                    payload_radius=self.r_payload,
                    drone_radius=self.drone_radius,
                    current_step=self.step_counters[0].item()
                )
            except Exception as e:
                print(f"Visualization error: {e}")
    def _record_payload_trajectories(self):
        """记录第一个环境的负载轨迹"""
        self.payload_traj_list[0].append(self.payload.state[0, 0:3].cpu().numpy())
        # 同步记录四元数，便于在终点正确绘制负载圆与挂载点
        self.payload_quat_traj_list[0].append(self.payload.state[0, 6:10].cpu().numpy())

    def draw_payload_traj_list(self):
        """绘制第一个环境的负载3D轨迹并保存到results/payload目录
           更新：
           - 负载用一个圆表示（圆在负载自身局部 XY 平面上，随四元数旋转）
           - 绳子挂载点使用 self.r_i（负载局部坐标），在图中绘制为点
           - 无人机用圆表示，半径由 self.drone_radius 指定（默认 0.125）
           - 在绘制完 3D 图后，额外绘制一张俯视图（XY 平面投影），负载/无人机/障碍物用不同颜色圆圈
           - 两张图片都保存到 results/payload 目录，文件名带时间戳
        """
        if not self.payload_traj_list or len(self.payload_traj_list[0]) == 0:
            print("No trajectory data to plot for environment 0")
            return

        # 创建保存目录
        save_dir = "results/payload"
        os.makedirs(save_dir, exist_ok=True)

        # ----- 3D 图 -----
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 只绘制第一个环境的轨迹
        env_idx = 0
        if len(self.payload_traj_list[env_idx]) > 1:  # 至少需要2个点才能画轨迹
            traj = np.array(self.payload_traj_list[env_idx])  # (n_points, 3)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='blue',
                    label=f'Actual Trajectory',
                    linewidth=2,
                    alpha=0.8)
            
            # 每隔10个点用紫色点标注
            ax.scatter(traj[::10, 0], traj[::10, 1], traj[::10, 2],
                       color='purple', marker='o', s=50, alpha=0.8, label='Marker (every 10 pts)')

            # 标记起点和终点
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                       color='green', marker='o', s=100, alpha=0.9, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                       color='red', marker='s', s=100, alpha=0.9, label='End')
            # 终点到地面的垂线
            end_x, end_y, end_z = traj[-1]
            ax.plot([end_x, end_x], [end_y, end_y], [0, end_z],
                    color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # 绘制负载圆与挂载点
        attach_world = None
        p_load = None
        try:
            # 使用轨迹的终点状态以避免重置后中心与轨迹终点不重合
            traj_np = np.array(self.payload_traj_list[0])
            p_load = traj_np[-1]
            # 若记录了四元数，则使用终点四元数，否则退化为单位旋转
            if len(self.payload_quat_traj_list[0]) > 0:
                q_end = torch.from_numpy(self.payload_quat_traj_list[0][-1]).unsqueeze(0)
                R_l = quat_to_rot(q_end).squeeze(0).cpu().numpy()
            else:
                R_l = np.eye(3)
            circle_radius = float(self.r_payload) if hasattr(self, 'r_payload') and self.r_payload is not None else 0.2
            theta = np.linspace(0, 2*np.pi, 60)
            circle_local = np.stack([circle_radius*np.cos(theta),
                                     circle_radius*np.sin(theta),
                                     np.zeros_like(theta)], axis=1)
            circle_world = (R_l @ circle_local.T).T + p_load
            ax.plot(circle_world[:,0], circle_world[:,1], circle_world[:,2],
                    color='magenta', linewidth=2, alpha=0.9, label='Payload (circle)')
            try:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                verts = [list(circle_world)]
                poly = Poly3DCollection(verts, alpha=0.15, facecolor='magenta', edgecolor=None)
                ax.add_collection3d(poly)
            except Exception:
                pass

            # 挂载点（局部 r_i -> 世界坐标）
            r_i_local = self.r_i.cpu().numpy()
            attach_world = (R_l @ r_i_local.T).T + p_load
            ax.scatter(attach_world[:,0], attach_world[:,1], attach_world[:,2],
                       color='red', marker='o', s=80, label='Attachment points', edgecolors='black')
            for aw in attach_world:
                ax.plot([p_load[0], aw[0]], [p_load[1], aw[1]], [p_load[2], aw[2]],
                        color='red', linestyle='--', linewidth=1.0, alpha=0.6)
        except Exception as e:
            print(f"Error drawing payload circle or attachment points: {e}")
            p_load = None

        # 参考轨迹（3D）
        if hasattr(self, 'ref_position') and self.ref_position is not None:
            ref_traj = np.array(self.ref_position)
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
                    'k--', linewidth=2, alpha=0.6, label='Reference Trajectory')
            # 参考轨迹上每隔10个点用紫色点标注
            ax.scatter(ref_traj[::10, 0], ref_traj[::10, 1], ref_traj[::10, 2],
                       color='purple', marker='^', s=50, alpha=0.8)
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], np.zeros_like(ref_traj[:, 2]),
                    color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
                    label='Ref Trajectory Projection')
            
            # 标注终止时的参考轨迹点位置
            current_ref_idx = min(self.current_point_indices[0].item(), len(ref_traj) - 1)
            ref_point = ref_traj[current_ref_idx]
            ax.scatter(ref_point[0], ref_point[1], ref_point[2],
                       color='cyan', marker='*', s=300, alpha=1.0, 
                       edgecolors='black', linewidths=2, label='Current Ref Point')
            ax.text(ref_point[0], ref_point[1], ref_point[2] + 0.15,
                    f"Ref[{current_ref_idx}]",
                    color='cyan', fontsize=9, weight='bold', ha='center')

        # 绘制无人机（3D 圆盘）
        drone_circle_radius = float(self.drone_radius) if hasattr(self, 'drone_radius') else 0.125
        if hasattr(self, 'drone_pos') and self.drone_pos is not None:
            try:
                drone_positions = self.drone_pos[0].cpu().numpy()
                drone_color = 'orange'
                theta = np.linspace(0, 2*np.pi, 40)
                for i, pos in enumerate(drone_positions):
                    circle_xy = np.stack([drone_circle_radius * np.cos(theta) + pos[0],
                                          drone_circle_radius * np.sin(theta) + pos[1],
                                          np.full_like(theta, pos[2])], axis=1)
                    ax.plot(circle_xy[:,0], circle_xy[:,1], circle_xy[:,2],
                            color=drone_color, linewidth=2, alpha=0.9,
                            label='Drones' if i == 0 else None)
                    try:
                        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                        verts = [list(circle_xy)]
                        poly = Poly3DCollection(verts, alpha=0.2, facecolor=drone_color, edgecolor=None)
                        ax.add_collection3d(poly)
                    except Exception:
                        pass

                    if attach_world is not None and i < attach_world.shape[0]:
                        aw = attach_world[i]
                        ax.plot([pos[0], aw[0]],
                                [pos[1], aw[1]],
                                [pos[2], aw[2]],
                                color=drone_color, linestyle='-', linewidth=1.5, alpha=0.6)
                    else:
                        if p_load is not None:
                            ax.plot([pos[0], p_load[0]],
                                    [pos[1], p_load[1]],
                                    [pos[2], p_load[2]],
                                    color=drone_color, linestyle='-', linewidth=1.5, alpha=0.6)
            except Exception as e:
                print(f"Error drawing drones: {e}")

        # 障碍物（3D）
        if len(self.obstacle_pos) > 0:
            for i, obs_pos in enumerate(self.obstacle_pos.cpu().numpy()):
                ax.plot([obs_pos[0], obs_pos[0]],
                        [obs_pos[1], obs_pos[1]],
                        [0, 3],
                        'r-', linewidth=4, alpha=0.8,
                        label=f'Obstacle {i+1}' if i == 0 else "")
                theta = np.linspace(0, 2*np.pi, 50)
                x_circle = obs_pos[0] + self.obstacle_r * np.cos(theta)
                y_circle = obs_pos[1] + self.obstacle_r * np.sin(theta)
                z_circle = np.zeros_like(x_circle)
                ax.plot(x_circle, y_circle, z_circle, 'r-', alpha=0.5, linewidth=1)
                z_circle_top = np.full_like(x_circle, 3)
                ax.plot(x_circle, y_circle, z_circle_top, 'r-', alpha=0.5, linewidth=1)

        # 3D 图属性 & 保存
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        traj_progress_value = self.traj_progress[0].item()
        env1_Reward = self.reward[0].item()
        common_title = (
            f'Payload Trajectory - Environment 0\n'
            f'Steps: {self.step_counters[0].item()} | '
            f'Progress: {self.achieve_num[0].item()}'
            f' | Reward: {env1_Reward:.2f} | Env1 Done: {self.env_done[0]}'
        )
        ax.set_title(common_title, fontsize=14)
        
        # 获取当前参考轨迹索引和期望状态
        current_ref_idx = min(self.current_point_indices[0].item(), len(self.ref_traj['Ref_xl']) - 1)
        ref_state = self.ref_traj['Ref_xl'][current_ref_idx]
        
        # 获取距离信息
        # 负载离障碍物距离（已在_check_done中计算）
        payload_to_obs_dist = self.payload_obstacle_min_dist[0].item() - self.r_payload - self.obstacle_r
        
        # 无人机离障碍物距离（最小值，已在_check_done中计算）
        if hasattr(self, 'min_drone_obstacle_dist') and self.min_drone_obstacle_dist is not None:
            drone_to_obs_dist = self.min_drone_obstacle_dist[0].min().item() - self.drone_radius - self.obstacle_r
        else:
            drone_to_obs_dist = 0.0
        
        # 无人机之间最小间距（已在_check_done中计算）
        drone_to_drone_dist = self.min_drone_drone_dist[0].item() if hasattr(self, 'min_drone_drone_dist') else 0.0
        
        # 负载和绳子状态作为文本注释
        payload_state = self.payload.state[0]
        cable_tensions = self.cable.T[0]
        
        state_text_2d = (
            f"Actual Payload State:\n"
            f"Pos: [{payload_state[0].item():.2f}, {payload_state[1].item():.2f}, {payload_state[2].item():.2f}]\n"
            f"Vel: [{payload_state[3].item():.2f}, {payload_state[4].item():.2f}, {payload_state[5].item():.2f}]\n"
            f"Quat: [{payload_state[6].item():.2f}, {payload_state[7].item():.2f}, {payload_state[8].item():.2f}, {payload_state[9].item():.2f}]\n"
            f"Omega: [{payload_state[10].item():.2f}, {payload_state[11].item():.2f}, {payload_state[12].item():.2f}]\n\n"
            f"Ref Payload State [idx={current_ref_idx}]:\n"
            f"Pos: [{ref_state[0]:.2f}, {ref_state[1]:.2f}, {ref_state[2]:.2f}]\n"
            f"Vel: [{ref_state[3]:.2f}, {ref_state[4]:.2f}, {ref_state[5]:.2f}]\n"
            f"Quat: [{ref_state[6]:.2f}, {ref_state[7]:.2f}, {ref_state[8]:.2f}, {ref_state[9]:.2f}]\n"
            f"Omega: [{ref_state[10]:.2f}, {ref_state[11]:.2f}, {ref_state[12]:.2f}]\n\n"
            f"Cable Tensions: [{cable_tensions[0].item():.2f}, {cable_tensions[1].item():.2f}, {cable_tensions[2].item():.2f}, {cable_tensions[3].item():.2f}]\n\n"
            f"Distance Info:\n"
            f"Payload-Obs: {payload_to_obs_dist:.3f}m\n"
            f"Drone-Obs(min): {drone_to_obs_dist:.3f}m\n"
            f"Drone-Drone(min): {drone_to_drone_dist:.3f}m"
        )
        
        # 在图上添加状态文本，不使用图例
        ax.text2D(0.02, 0.98, state_text_2d, transform=ax.transAxes,
                  fontsize=7, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.grid(True)

        ax.set_xlim([-1,3])
        ax.set_ylim([-1,3])
        ax.set_zlim([0, 4])

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        filename_3d = f"{timestamp}.png"
        filepath_3d = os.path.join(save_dir, filename_3d)

        plt.tight_layout()
        plt.savefig(filepath_3d, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Environment 0 payload trajectory saved to: {filepath_3d}")

        # 奖励不在图中额外标注，已置于图例标题

        # ----- 俯视图（Top-down XY） -----
        if not ENABLE_TOP_VIEW:
            # 清空轨迹记录
            self.payload_traj_list[0].clear()
            self.payload_quat_traj_list[0].clear()
            print("Environment 0 trajectory record cleared (top view disabled)")
            return

        try:
            fig2, ax2 = plt.subplots(figsize=(8, 8))

            # 轨迹投影
            if len(self.payload_traj_list[env_idx]) > 0:
                traj_xy = np.array(self.payload_traj_list[env_idx])[:, :2]
                ax2.plot(traj_xy[:, 0], traj_xy[:, 1], color='blue', linewidth=2, label='Actual Trajectory (proj)')
                # 每隔10个点用紫色点标注
                ax2.scatter(traj_xy[::10, 0], traj_xy[::10, 1], color='purple', s=40, label='Marker (every 10 pts)')
                ax2.scatter(traj_xy[0, 0], traj_xy[0, 1], color='green', s=60, label='Start (proj)')
                ax2.scatter(traj_xy[-1, 0], traj_xy[-1, 1], color='red', s=60, label='End (proj)')

            # 参考轨迹投影
            if hasattr(self, 'ref_position') and self.ref_position is not None:
                ref_traj = np.array(self.ref_position)
                ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'k--', linewidth=1.5, alpha=0.7, label='Ref Trajectory (proj)')
                # 参考轨迹上每隔10个点用紫色点标注
                ax2.scatter(ref_traj[::10, 0], ref_traj[::10, 1], color='purple', marker='^', s=40, alpha=0.8)
                
                # 标注终止时的参考轨迹点位置（2D投影）
                current_ref_idx_2d_marker = min(self.current_point_indices[0].item(), len(ref_traj) - 1)
                ref_point_2d = ref_traj[current_ref_idx_2d_marker]
                ax2.scatter(ref_point_2d[0], ref_point_2d[1],
                           color='cyan', marker='*', s=300, alpha=1.0,
                           edgecolors='black', linewidths=2, label='Current Ref Point')
                ax2.text(ref_point_2d[0], ref_point_2d[1] + 0.1,
                        f"Ref[{current_ref_idx_2d_marker}]\n({ref_point_2d[0]:.2f}, {ref_point_2d[1]:.2f})",
                        color='cyan', fontsize=9, weight='bold', ha='center')

            # 负载（用圆表示）- 使用终点中心
            if len(self.payload_traj_list[env_idx]) > 0:
                end_xy = np.array(self.payload_traj_list[env_idx])[-1][:2]
                load_circle = plt.Circle((end_xy[0], end_xy[1]), circle_radius, color='magenta', alpha=0.25, label='Payload (proj)')
                ax2.add_patch(load_circle)
                ax2.scatter(end_xy[0], end_xy[1], color='magenta', s=40)

            # 无人机（圆）
            if hasattr(self, 'drone_pos') and self.drone_pos is not None:
                drone_positions = self.drone_pos[0].cpu().numpy()
                for i, pos in enumerate(drone_positions):
                    c = plt.Circle((pos[0], pos[1]), drone_circle_radius, color='orange', alpha=0.4, label='Drones (proj)' if i == 0 else None)
                    ax2.add_patch(c)
                    ax2.scatter(pos[0], pos[1], color='orange', edgecolors='black', s=30)

            # 障碍物（圆）
            if len(self.obstacle_pos) > 0:
                for i, obs_pos in enumerate(self.obstacle_pos.cpu().numpy()):
                    c_obs = plt.Circle((obs_pos[0], obs_pos[1]), self.obstacle_r, color='red', alpha=0.25, label='Obstacles' if i == 0 else None)
                    ax2.add_patch(c_obs)
                    ax2.scatter(obs_pos[0], obs_pos[1], color='red', s=30)

            # 绘制挂载点投影
            if attach_world is not None:
                aw_xy = attach_world[:, :2]
                ax2.scatter(aw_xy[:, 0], aw_xy[:, 1], color='red', s=30, marker='x', label='Attachment (proj)')

            # 格式化轴
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_aspect('equal', adjustable='box')
            ax2.grid(True)
            # 统一标题与3D图一致
            ax2.set_title(common_title)

            # 设定显示范围，尽量包络轨迹和对象
            try:
                all_x = []
                all_y = []
                if len(self.payload_traj_list[env_idx]) > 0:
                    all_x.extend(traj_xy[:, 0].tolist()); all_y.extend(traj_xy[:, 1].tolist())
                if hasattr(self, 'ref_position') and self.ref_position is not None:
                    all_x.extend(ref_traj[:, 0].tolist()); all_y.extend(ref_traj[:, 1].tolist())
                if p_load is not None:
                    all_x.append(p_load[0]); all_y.append(p_load[1])
                if hasattr(self, 'drone_pos') and self.drone_pos is not None:
                    all_x.extend(drone_positions[:, 0].tolist()); all_y.extend(drone_positions[:, 1].tolist())
                if len(self.obstacle_pos) > 0:
                    obs_np = self.obstacle_pos.cpu().numpy()
                    all_x.extend(obs_np[:, 0].tolist()); all_y.extend(obs_np[:, 1].tolist())

                if len(all_x) > 0:
                    margin = 0.5
                    ax2.set_xlim(min(all_x) - margin, max(all_x) + margin)
                    ax2.set_ylim(min(all_y) - margin, max(all_y) + margin)
            except Exception:
                pass

            # 负载和绳子状态显示（与3D图一致）
            payload_state = self.payload.state[0]
            cable_tensions = self.cable.T[0]
            
            # 获取当前参考轨迹索引和期望状态（复用3D图的逻辑）
            current_ref_idx_2d = min(self.current_point_indices[0].item(), len(self.ref_traj['Ref_xl']) - 1)
            ref_state_2d = self.ref_traj['Ref_xl'][current_ref_idx_2d]
            
            # 获取距离信息
            payload_to_obs_dist_2d = self.payload_obstacle_min_dist[0].item() - self.r_payload - self.obstacle_r
            if hasattr(self, 'min_drone_obstacle_dist') and self.min_drone_obstacle_dist is not None:
                drone_to_obs_dist_2d = self.min_drone_obstacle_dist[0].min().item() - self.drone_radius - self.obstacle_r
            else:
                drone_to_obs_dist_2d = 0.0
            drone_to_drone_dist_2d = self.min_drone_drone_dist[0].item() if hasattr(self, 'min_drone_drone_dist') else 0.0
            
            state_text_2d = (
                f"Actual Payload State:\n"
                f"Pos: [{payload_state[0].item():.2f}, {payload_state[1].item():.2f}, {payload_state[2].item():.2f}]\n"
                f"Vel: [{payload_state[3].item():.2f}, {payload_state[4].item():.2f}, {payload_state[5].item():.2f}]\n"
                f"Quat: [{payload_state[6].item():.2f}, {payload_state[7].item():.2f}, {payload_state[8].item():.2f}, {payload_state[9].item():.2f}]\n"
                f"Omega: [{payload_state[10].item():.2f}, {payload_state[11].item():.2f}, {payload_state[12].item():.2f}]\n\n"
                f"Ref Payload State [idx={current_ref_idx_2d}]:\n"
                f"Pos: [{ref_state_2d[0]:.2f}, {ref_state_2d[1]:.2f}, {ref_state_2d[2]:.2f}]\n"
                f"Vel: [{ref_state_2d[3]:.2f}, {ref_state_2d[4]:.2f}, {ref_state_2d[5]:.2f}]\n"
                f"Quat: [{ref_state_2d[6]:.2f}, {ref_state_2d[7]:.2f}, {ref_state_2d[8]:.2f}, {ref_state_2d[9]:.2f}]\n"
                f"Omega: [{ref_state_2d[10]:.2f}, {ref_state_2d[11]:.2f}, {ref_state_2d[12]:.2f}]\n\n"
                f"Cable Tensions: [{cable_tensions[0].item():.2f}, {cable_tensions[1].item():.2f}, {cable_tensions[2].item():.2f}, {cable_tensions[3].item():.2f}]\n\n"
                f"Distance Info:\n"
                f"Payload-Obs: {payload_to_obs_dist_2d:.3f}m\n"
                f"Drone-Obs(min): {drone_to_obs_dist_2d:.3f}m\n"
                f"Drone-Drone(min): {drone_to_drone_dist_2d:.3f}m"
            )
            
            # 在图上添加状态文本，放在右下角
            ax2.text(0.98, 0.02, state_text_2d, transform=ax2.transAxes,
                     fontsize=7, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            filename_2d = f"payload_trajectory_env0_topdown_{timestamp}_steps{self.step_counters[0].item()}.png"
            filepath_2d = os.path.join(save_dir, filename_2d)
            plt.tight_layout()
            plt.savefig(filepath_2d, dpi=300, bbox_inches='tight')
            plt.close()   
            print(f"Environment 0 top-down view saved to: {filepath_2d}")
        except Exception as e:
            print(f"Failed to create top-down view: {e}")

        # 清空轨迹记录
        self.payload_traj_list[0].clear()
        self.payload_quat_traj_list[0].clear()
        print("Environment 0 trajectory record cleared")