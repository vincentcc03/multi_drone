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

# 控制print输出的开关
ENABLE_PRINT = False  # 设置为False即可关闭所有print

# 控制可视化的开关
ENABLE_VISUALIZATION = True  # 设置为False即可关闭所有可视化

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
        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch()
        self.cable = CableDynamicsSimBatch()
        self.v_dot = self.payload.v_dot 
        self.omega_dot = self.payload.omega_dot
        
        # 奖励log
        self.reward_log = torch.zeros(4, dtype=torch.float32, device=self.device)
        # 参考轨迹
        self.ref_traj = generate_complete_trajectory()
        self.payload_init_state = torch.from_numpy(self.ref_traj['Ref_xl'][0]).float()
        print("Payload initial state:", self.payload_init_state)
        # 轨迹进展
        self.traj_progress = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        
        # 可视化
        self.visualizer = DroneVisualization()
        self.ref_traj_vis = self.ref_traj['Ref_xl'][:, 0:3]  # 仅位置部分
        self.payload_traj_list = []
        for _ in range(self.num_envs):
            self.payload_traj_list.append([])
        
        
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
        self.interval = self.traj_config["dt"]/self.config["dt"]
        

        # obs / act 空间
        self.obs_dim = 39
        self.act_dim = 4 * self.n_cables
        
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
            self.logger.log_step(self.step_counters.max().item(), self.reward.max().item(), self.reward_log[0].item(), self.reward_log[1].item(), self.reward_log[2].item(), self.reward_log[3].item())
            # 向量化重置（性能更好）
            if reset_mask[0].item():  # 如果第一个环境需要重置，清空其轨迹记录器
                self.draw_payload_traj_list()
            num_reset = reset_mask.sum().item()
            
            # 扩展初始状态到需要重置的环境数量
            payload_init_expanded = self.payload_init_state.unsqueeze(0).expand(num_reset, -1).to(self.device)
            cable_init_expanded = self.cable_init_state.unsqueeze(0).expand(num_reset, -1, -1).to(self.device)
            
            # 批量重置
            self.payload.state[reset_mask] = payload_init_expanded
            self.cable.state[reset_mask] = cable_init_expanded
            self.reward[reset_mask] = 0.0
            self.pos_error_count[reset_mask] = 0
            # 重置计数器
            # print("done:",reset_mask)
            
            self.current_point_indices[reset_mask] = 1
            self.step_counters[reset_mask] = 0
            
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
        action *= 5
        # print("Action:", action)

        # 动力学仿真
        self.cable.rk4_step(action)
        
        q_l = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)
        # print("Input force and torque to payload:", input_force_torque)
        self.payload.rk4_step(input_force_torque)
        
        # print("Payload state:", self.payload.state)
        # 计算无人机位置
        self.compute_drone_positions()
        
        

        # 奖励 + done
        self.done , info = self._check_done()
        reward = self._compute_reward(action)
        self.reward += reward
        print("total_rewards:", self.reward)
        max_reward = self.reward.max().item()
        print(f"max reward: {max_reward}")
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
        current_time = self.step_counters[0].item() * self.dt
        # print(f"Current sim time: {current_time:.2f}s")
        if ENABLE_VISUALIZATION and current_time*100 % 4 == 0:
            print(f"Visualizing at sim time: {current_time:.2f}s")
            # self._visualize_current_state()
            self._record_payload_trajectories()

        obs = self._get_obs()

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

        # 参考状态 - 为每个环境使用独立的轨迹索引
        ref_xl_batch = torch.zeros(B, 13, dtype=torch.float32, device=self.device)
        for i in range(B):
            next_idx = min(self.current_point_indices[i].item() + 1, len(self.ref_traj['Ref_xl']) - 1)
            ref_xl_batch[i] = torch.tensor(self.ref_traj['Ref_xl'][next_idx], dtype=torch.float32, device=self.device)
        
        p_l_ref = ref_xl_batch[:, 0:3]
        v_l_ref = ref_xl_batch[:, 3:6]
        q_l_ref = ref_xl_batch[:, 6:10]
        omega_l_ref = ref_xl_batch[:, 10:13]

        # 误差计算
        pos_error = torch.norm(p_l - p_l_ref, dim=1)
        vel_error = torch.norm(v_l - v_l_ref, dim=1)
        # 四元数误差计算
        cos_half_theta = torch.abs(torch.sum(q_l * q_l_ref, dim=1))
        cos_half_theta = torch.clamp(cos_half_theta, -1.0, 1.0)
        delta_theta = 2.0 * torch.acos(cos_half_theta)  # 弧度误差 ∈ [0, π]
        quat_error = delta_theta
        
        omega_error = torch.norm(omega_l - omega_l_ref, dim=1)
        # print ("p_l_ref:", p_l_ref)
        # print(f"Position errors: {pos_error}")
        # print(f"Velocity errors: {vel_error}")
        # print(f"Quaternion errors: {quat_error}")
        # print(f"Omega errors: {omega_error}")

        dt = self.config.get("dt")  # 仿真步长
        
        # 位置奖励：指数衰减
        pos_scale = self.config.get("pos_reward_scale")  # 距离缩放系数
        reward_pos = -self.pos_w * pos_error * dt * pos_scale
        
        # 速度奖励：指数衰减
        vel_scale = self.config.get("vel_reward_scale")
        reward_vel = -self.vel_w * vel_error * dt * vel_scale

        # 姿态奖励：指数衰减
        quat_scale = self.config.get("quat_reward_scale")  # 四元数误差通常较小
        reward_quat = -self.quat_w * quat_error * dt * quat_scale

        # 角速度奖励：指数衰减
        omega_scale = self.config.get("omega_reward_scale")
        reward_omega = -self.omega_w * omega_error * dt * omega_scale

        # 总奖励：正值，鼓励减小误差
        self.reward_log[0] = reward_pos.mean().item()
        self.reward_log[1] = reward_vel.mean().item()
        self.reward_log[2] = reward_quat.mean().item()
        self.reward_log[3] = reward_omega.mean().item()
        reward = reward_pos + reward_vel + reward_quat + reward_omega
        
        
        # # 进度奖励

        # progress_reach = (self.current_point_indices > 20) & (pos_error < 0.05)
        # reward += progress_reach.float() * 0.1

        # # 终止惩罚
        # reward += self.config.get("done_penalty", -1.0) * self.done.float()
        # 可选：添加额外奖励项
        # 1. 到达目标点的奖励
        # if pos_error.min() < 0.1:  # 到达目标附近
        #     reward += 100 * dt
    
        # 2. 轨迹进展奖励
        # traj_progress = self.current_point_indices.float() / 100
        # 动作平滑奖励
        # action_smoothness = -self.payload.v_dot.norm(dim=1) - self.payload.omega_dot.norm(dim=1)
        # progress_reward = self.config.get("progress_reward", 1)
        # reward += progress_reward * dt
        # reward += action_smoothness * dt
        # print(f"Progress reward: {progress_reward * dt}，action_smoothness: {action_smoothness.mean().item()*dt:.4f}")
        # reward += traj_progress * dt
        # print(f"Progress reward: {progress_reward * dt}, Traj progress: {(traj_progress * dt).mean().item():.4f}")
        print(f"Reward components : {reward_pos.mean().item():.6f}, {reward_vel.mean().item():.6f}, {reward_quat.mean().item():.6f}, {reward_omega.mean().item():.6f}")
        
        return reward

    def _check_done(self):
        B = self.num_envs
        # 每个环境一个 done 标志
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        # # 1. 位置误差终止条件 - 为每个环境使用独立的轨迹索引
        current_pos = self.payload.state[:, 0:3]   # (B, 3)
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
        
        done |= (payload_height <= 0.0)
        if (payload_height <= 0.0).any():
            print("payload heights:", payload_height)
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
            min_dist, _ = torch.min(dist_to_obs, dim=1)            # (B,)
            done |= (min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance))
            if (min_dist < (self.r_payload + self.obstacle_r + self.collision_tolerance)).any():
                print("Min distances from payload to obstacles:", min_dist)
            # 无人机与障碍物碰撞检测
            drone_xy = self.drone_pos[:, :, :2].unsqueeze(2)   # (B, N_cables, 1, 2)
            obs_pos = obs_pos.unsqueeze(1)                     # (B, 1, N, 2)
            dist_to_obs = torch.norm(drone_xy - obs_pos, dim=3)  # (B, N_cables, N)
            min_dist, _ = torch.min(dist_to_obs, dim=2)          # (B, N_cables)
            done |= (min_dist < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any(dim=1)
            if (min_dist < (self.drone_radius + self.obstacle_r + self.collision_tolerance)).any():
                print("Min distances from drones to obstacles:", min_dist)
            # if done.all():
                # print("Min distances from drones to obstacles:")
                
            # # 无人机之间的碰撞检测
            # drone_xy = self.drone_pos[:, :, :2]  # (B, N_cables, 2)
            # drone_z = self.drone_pos[:, :, 2]    # (B, N_cables)
            # for i in range(self.n_cables):
            #     for j in range(i+1, self.n_cables):
            #         # xy平面距离
            #         dist_xy = torch.norm(drone_xy[:, i, :] - drone_xy[:, j, :], dim=1)  # (B,)
            #         # z轴距离
            #         dist_z = torch.abs(drone_z[:, i] - drone_z[:, j])  # (B,)
            #         # xy距离和z距离都小于阈值才算碰撞
            #         collision = (dist_xy < (2*self.drone_radius + self.collision_tolerance)) & \
            #                     (dist_z < self.drone_high_distance)
            #         done |= collision
            #         if done.all():
            #             print("Drone collisions:")
                    
        # 4. 绳子拉力终止条件
        done |= (self.cable.T > self.t_max).any(dim=1)
        if (self.cable.T > self.t_max).any():
            print("Max cable tensions:", self.cable.T.max(dim=1).values)
        # if done.all():
            # print("Max cable tensions:")
        # 时间终止条件
        done |= (self.current_point_indices >= len(self.ref_traj['Ref_xl']) - 1)
        # 离终点距离最大偏移终止条件
        dis_to_end = torch.norm(current_pos - torch.tensor(self.ref_traj['Ref_xl'][-1][0:3], device=self.device), dim=1)
        done |= (dis_to_end > 5.0)
        info = [{} for _ in range(B)]
        for i in range(B):
            if done[i]:
                info[i]["episode"] = {
                    "step": self.step_counters[i].item(),
                    "total_reward": self.reward[i].item()  # 注意这里要用累计奖励
                }
        return done, info

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

        # 3. ref_xl (B, 13) 和 ref_ul (B, 18) - 为每个环境使用独立的轨迹索引
        ref_xl_batch = torch.zeros(B, 13, dtype=torch.float32, device=self.device)
        ref_ul_batch = torch.zeros(B, 18, dtype=torch.float32, device=self.device)
        
        for i in range(B):
            next_idx = min(self.current_point_indices[i].item() + 1, len(self.ref_traj['Ref_xl']) - 1)
            curr_idx = min(self.current_point_indices[i].item(), len(self.ref_traj['Ref_ul']) - 1)
            
            ref_xl_batch[i] = torch.tensor(self.ref_traj['Ref_xl'][next_idx], dtype=torch.float32, device=self.device)
            ref_ul_batch[i] = torch.tensor(self.ref_traj['Ref_ul'][curr_idx], dtype=torch.float32, device=self.device)
        
        ref_xl = ref_xl_batch
        ref_ul = ref_ul_batch

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
                    current_step=self.step_counters[0].item()
                )
            except Exception as e:
                print(f"Visualization error: {e}")
    def _record_payload_trajectories(self):
        """记录第一个环境的负载轨迹"""
        self.payload_traj_list[0].append(self.payload.state[0, 0:3].cpu().numpy())

    def draw_payload_traj_list(self):
        """绘制第一个环境的负载3D轨迹并保存到results/payload目录"""
        if not self.payload_traj_list or len(self.payload_traj_list[0]) == 0:
            print("No trajectory data to plot for environment 0")
            return
        
        # 创建保存目录
        save_dir = "results/payload"
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建3D图形
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
            
            # 标记起点和终点
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                    color='green', marker='o', s=100, alpha=0.9, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                    color='red', marker='s', s=100, alpha=0.9, label='End')
            
            print(f"绘制轨迹：{len(traj)} 个点")
        
        # 绘制参考轨迹
        if hasattr(self, 'ref_traj_vis') and self.ref_traj_vis is not None:
            ref_traj = np.array(self.ref_traj_vis)
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 
                'k--', linewidth=2, alpha=0.6, label='Reference Trajectory')
        
        # 绘制障碍物 - 使用竖直线表示
        if len(self.obstacle_pos) > 0:
            for i, obs_pos in enumerate(self.obstacle_pos.cpu().numpy()):
                # 绘制障碍物中心的竖直线
                ax.plot([obs_pos[0], obs_pos[0]], 
                    [obs_pos[1], obs_pos[1]], 
                    [0, 3], 
                    'r-', linewidth=4, alpha=0.8, 
                    label=f'Obstacle {i+1}' if i == 0 else "")
                
                # 在底部绘制圆圈表示障碍物边界
                theta = np.linspace(0, 2*np.pi, 50)
                x_circle = obs_pos[0] + self.obstacle_r * np.cos(theta)
                y_circle = obs_pos[1] + self.obstacle_r * np.sin(theta)
                z_circle = np.zeros_like(x_circle)  # 在地面画圆
                ax.plot(x_circle, y_circle, z_circle, 'r-', alpha=0.5, linewidth=1)
                
                # 在顶部也画一个圆
                z_circle_top = np.full_like(x_circle, 3)
                ax.plot(x_circle, y_circle, z_circle_top, 'r-', alpha=0.5, linewidth=1)
        
        # 设置图形属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Payload Trajectory - Environment 0 (Steps: {self.step_counters[0].item()})')
        ax.legend()
        ax.grid(True)
        
        # 设置相等的坐标轴比例
        max_range = 2.0  # 设置默认范围
        if len(self.payload_traj_list[0]) > 0:
            traj = np.array(self.payload_traj_list[0])
            traj_range = np.max(np.abs(traj))
            max_range = max(max_range, traj_range)
        
        # 同时考虑参考轨迹的范围
        if hasattr(self, 'ref_traj_vis') and self.ref_traj_vis is not None:
            ref_range = np.max(np.abs(self.ref_traj_vis))
            max_range = max(max_range, ref_range)
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"payload_trajectory_env0_{timestamp}_steps{self.step_counters[0].item()}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        
        print(f"Environment 0 payload trajectory saved to: {filepath}")
        
        # 只清空第一个环境的轨迹记录器
        self.payload_traj_list[0].clear()
        
        print("Environment 0 trajectory record cleared")