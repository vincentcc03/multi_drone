import gym
import numpy as np
import torch
from src.utils.read_npy import load_trajectory_point
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.envs.dynamics.rope_dynamic import CableDynamicsSimBatch
from src.utils.computer import quat_to_rot
from src.utils.read_yaml import load_config

class Env(gym.Env):
    def __init__(self, batch_size=1, device=None, n_cables=4):
        super().__init__()
        self.config = load_config("env_config.yaml")
        self.device = device if device is not None else torch.device(self.config.get("device", "cpu"))
        self.batch_size = self.config.get("batch_size", 1)
        self.n_cables = self.config.get("rope_num", 4)

        # 动力学仿真器
        self.payload = PayloadDynamicsSimBatch(config_path="env_config.yaml")
        self.cable = CableDynamicsSimBatch(config_path="env_config.yaml")
        
        # 从配置文件获取参数
        self.dt = self.config.get("dt", 0.01)
        self.rope_length = self.config.get("rope_length", 1.0)
        self.quad_mass = self.config.get("quad_mass", 1.0)
        self.m_l = self.config.get("m_l", 1.0)
        self.g = self.config.get("g", 9.81)
        
        # 绳子挂载点
        self.di_list = torch.tensor(self.config.get("di_list"), 
                                   dtype=torch.float32, device=self.device)
        
        # 障碍物信息
        self.obstacle_pos = torch.tensor(self.config.get("obstacle_pos"), 
                                       dtype=torch.float32, device=self.device)
        self.obstacle_r = self.config.get("obstacle_r")
        
        # 安全约束参数
        self.d_min_quad = self.config.get("d_min_quad", 1.0)
        self.d_min_obs = self.config.get("d_min_obs", 1.5)
        self.t_max = self.config.get("t_max", 50.0)
        self.f_max = self.config.get("f_max", 100.0)
        
        # 奖励权重
        self.tracking_weight = self.config.get("tracking_weight", 1.0)
        self.control_cost_weight = self.config.get("control_cost_weight", 0.01)
        self.velocity_penalty_weight = self.config.get("velocity_penalty_weight", 0.001)
        self.orientation_penalty_weight = self.config.get("orientation_penalty_weight", 0.1)
        self.constraint_penalty_weight = self.config.get("constraint_penalty_weight", 100.0)
        
        # 终止条件
        self.min_height = self.config.get("min_height", 0.5)
        self.max_tracking_error = self.config.get("max_tracking_error", 5.0)
        self.collision_tolerance = self.config.get("collision_tolerance", 0.1)
        
        # PPO相关参数
        self.total_timesteps = self.config.get("total_timesteps", 1000000)
        self.n_steps = self.config.get("n_steps", 2048)
        self.trajectory_length = self.config.get("trajectory_length")

        # 计算轨迹点更新频率
        self.points_per_epoch = max(1, self.trajectory_length // (self.total_timesteps // self.n_steps))
        self.current_point_index = 0
        self.step_counter = 0

        # observation = r_g(3) + obstacle(2*3=6) + 下一轨迹点(3) = 12维
        obs_dim = 3 + len(self.obstacle_pos) * 3 + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # action = n * 6 (angle, omega, gamma, tension, tension_dot, tension_ddot)
        act_dim = self.n_cables * 6
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(act_dim,), dtype=np.float32)

    def compute_quad_positions(self, payload_state=None, cable_state=None):
        """
        计算四旋翼位置
        根据公式: p_i = p_l + R_l * r_i + l_i * d_i
        其中：
        - p_l: 负载位置
        - R_l: 负载旋转矩阵  
        - r_i: 第i个挂载点在负载体坐标系中的位置
        - l_i: 第i根绳子长度
        - d_i: 第i根绳子方向单位向量
        """
        if payload_state is None:
            payload_state = self.payload.state
        if cable_state is None:
            cable_state = self.cable.state
            
        B = payload_state.shape[0]
        
        # 负载位置和姿态
        p_l = payload_state[:, 0:3]  # 负载位置
        q_l = payload_state[:, 6:10]  # 负载四元数
        R_l = quat_to_rot(q_l)  # 负载旋转矩阵 (B, 3, 3)
        
        quad_positions = []
        
        for i in range(self.n_cables):
            # 挂载点在负载体坐标系中的位置
            r_i = self.di_list[i].unsqueeze(0).expand(B, 3)  # (B, 3)
            
            # 挂载点在世界坐标系中的位置
            mount_point = p_l + torch.bmm(R_l, r_i.unsqueeze(-1)).squeeze(-1)
            
            # 绳子角度（假设在索引0）
            cable_angle = cable_state[:, i, 0]
            
            # 绳子方向单位向量 (假设在xy平面内摆动)
            d_i = torch.stack([
                torch.cos(cable_angle),
                torch.sin(cable_angle),
                torch.zeros_like(cable_angle)
            ], dim=1)  # (B, 3)
            
            # 四旋翼位置
            p_i = mount_point + self.rope_length * d_i
            quad_positions.append(p_i)
        
        return torch.stack(quad_positions, dim=1)  # (B, n_cables, 3)

    def reset(self):
        """
        重置环境状态，从YAML配置文件读取初始状态
        """
        B = self.batch_size
        
        # 1. 重置负载状态 (13维: position(3) + velocity(3) + quaternion(4) + angular_velocity(3))
        self.payload.state = torch.zeros(B, 13, dtype=torch.float32, device=self.device)
        
        # 从配置文件读取负载初始状态
        initial_pos = torch.tensor(self.config.get("payload_initial_position"), 
                                  dtype=torch.float32, device=self.device)
        initial_vel = torch.tensor(self.config.get("payload_initial_velocity"), 
                                  dtype=torch.float32, device=self.device)
        initial_quat = torch.tensor(self.config.get("payload_initial_quaternion"), 
                                   dtype=torch.float32, device=self.device)
        initial_omega = torch.tensor(self.config.get("payload_initial_angular_velocity"), 
                                    dtype=torch.float32, device=self.device)
        
        self.payload.state[:, 0:3] = initial_pos.unsqueeze(0).expand(B, 3)
        self.payload.state[:, 3:6] = initial_vel.unsqueeze(0).expand(B, 3)
        self.payload.state[:, 6:10] = initial_quat.unsqueeze(0).expand(B, 4)
        self.payload.state[:, 10:13] = initial_omega.unsqueeze(0).expand(B, 3)

        # 2. 重置绳子状态 (n_cables x 8维)
        self.cable.state = torch.zeros(B, self.n_cables, 8, dtype=torch.float32, device=self.device)
        
        # 从配置文件读取绳子初始状态
        initial_angles = torch.tensor(self.config.get("cable_initial_angles"), 
                                     dtype=torch.float32, device=self.device)
        initial_angular_vels = torch.tensor(self.config.get("cable_initial_angular_velocities"), 
                                           dtype=torch.float32, device=self.device)
        initial_tensions = torch.tensor(self.config.get("cable_initial_tensions"), 
                                       dtype=torch.float32, device=self.device)
        
        self.cable.state[:, :, 0] = initial_angles.unsqueeze(0).expand(B, self.n_cables)
        self.cable.state[:, :, 1] = initial_angular_vels.unsqueeze(0).expand(B, self.n_cables)
        self.cable.state[:, :, 2] = initial_tensions.unsqueeze(0).expand(B, self.n_cables)

        # 3. 重置轨迹跟踪
        self.current_point_index = 0
        self.step_counter = 0

        # 4. 获取初始观测
        obs = self._get_obs()
        return obs

    def step(self, action):
        B = self.batch_size
        action = torch.tensor(action, dtype=torch.float32, device=self.device).view(B, self.n_cables, 6)

        # 更新轨迹点索引（每隔一定步数更新）
        self.step_counter += 1
        if self.step_counter % self.points_per_epoch == 0:
            self.current_point_index = (self.current_point_index + 1) % self.trajectory_length

        # --- 拆分动作 ---
        angle = action[:, :, 0]
        omega = action[:, :, 1]
        gamma = action[:, :, 2]        # 角加速度
        tension = action[:, :, 3]
        tension_dot = action[:, :, 4]   # 张力变化率
        tension_ddot = action[:, :, 5]  # 张力二阶导数

        # 传递给绳子动力学的动作
        cable_action = torch.stack([gamma, tension_dot], dim=-1)

        # --- 更新绳子 ---
        self.cable.rk4_step(cable_action)

        # --- 计算合力合矩 ---
        q_l = self.payload.state[:, 6:10]
        R_l = quat_to_rot(q_l)
        input_force_torque = self.cable.compute_force_torque(R_l)

        # --- 更新负载 ---
        self.payload.rk4_step(input_force_torque)

        # --- 观测 ---
        obs = self._get_obs()

        # --- 计算奖励 ---
        reward, info = self._compute_reward(action)

        # --- 检查终止条件 ---
        done = self._check_done()

        return obs, reward, done, info

    def _compute_reward(self, action):
        """
        计算奖励函数，包含多个组成部分
        """
        B = self.batch_size
        
        # 1. 轨迹跟踪奖励
        current_pos = self.payload.state[:, 0:3]  # 当前负载位置
        target_pos = load_trajectory_point(self.current_point_index).unsqueeze(0).expand(B, 3)
        
        tracking_error = torch.norm(current_pos - target_pos, dim=1)
        tracking_reward = -self.tracking_weight * tracking_error
        
        # 2. 控制代价（惩罚过大的控制输入）
        control_cost = torch.norm(action, dim=(1, 2))
        control_penalty = -self.control_cost_weight * control_cost
        
        # 3. 约束违反惩罚
        constraint_penalty = self._compute_constraint_penalty()
        
        # 4. 速度惩罚（鼓励平滑运动）
        velocity = self.payload.state[:, 3:6]
        velocity_penalty = -self.velocity_penalty_weight * torch.norm(velocity, dim=1)
        
        # 5. 姿态稳定性奖励（鼓励负载保持水平）
        quaternion = self.payload.state[:, 6:10]
        z_world = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        R_l = quat_to_rot(quaternion)
        z_body = R_l[:, :, 2]  # 负载体坐标系z轴在世界坐标系中的方向
        orientation_error = 1.0 - torch.abs(torch.sum(z_body * z_world, dim=1))
        orientation_penalty = -self.orientation_penalty_weight * orientation_error
        
        # 总奖励
        total_reward = (tracking_reward + control_penalty + constraint_penalty + 
                       velocity_penalty + orientation_penalty).mean()
        
        # 信息字典
        info = {
            'tracking_error': tracking_error.mean().item(),
            'control_cost': control_cost.mean().item(),
            'constraint_penalty': constraint_penalty.mean().item(),
            'velocity_penalty': velocity_penalty.mean().item(),
            'orientation_penalty': orientation_penalty.mean().item(),
            'current_point_index': self.current_point_index,
            'target_position': target_pos[0].cpu().numpy(),
            'current_position': current_pos[0].cpu().numpy(),
        }
        
        return total_reward.item(), info

    def _compute_constraint_penalty(self):
        """
        计算约束违反惩罚，基于动力学方程中的约束
        """
        B = self.batch_size
        penalty = torch.zeros(B, device=self.device)
        
        # 计算四旋翼位置
        quad_positions = self.compute_quad_positions()  # (B, n_cables, 3)
        
        # 1. 四旋翼之间距离约束 (公式60a: ||p_i - p_j|| ≥ d_min^q)
        for i in range(self.n_cables):
            for j in range(i+1, self.n_cables):
                dist_ij = torch.norm(quad_positions[:, i] - quad_positions[:, j], dim=1)
                violation = torch.clamp(self.d_min_quad - dist_ij, min=0)
                penalty += self.constraint_penalty_weight * violation
        
        # 2. 四旋翼与障碍物距离约束 (公式60b: ||p_i - p_o|| ≥ d_min^o)
        for i in range(self.n_cables):
            for obs_pos in self.obstacle_pos:
                # 障碍物在3D空间中的位置（假设z坐标与负载相同）
                obs_pos_3d = torch.cat([obs_pos, self.payload.state[:, 2:3]], dim=0)
                dist_to_obs = torch.norm(quad_positions[:, i] - obs_pos_3d.unsqueeze(0), dim=1)
                violation = torch.clamp(self.d_min_obs - dist_to_obs, min=0)
                penalty += self.constraint_penalty_weight * violation
        
        # 3. 张力约束 (公式58: 0 < t_i ≤ t_max)
        tensions = self.cable.state[:, :, 2]  # 张力
        
        # 张力不能为负
        negative_tension = torch.clamp(-tensions, min=0)
        penalty += self.constraint_penalty_weight * torch.sum(negative_tension, dim=1)
        
        # 张力不能超过最大值
        excess_tension = torch.clamp(tensions - self.t_max, min=0)
        penalty += self.constraint_penalty_weight * torch.sum(excess_tension, dim=1)
        
        # 4. 推力约束 (公式62: ||m_i(p̈_i + ge_3) + t_i d_i|| ≤ f_max)
        for i in range(self.n_cables):
            # 计算四旋翼需要的推力
            # 简化处理：假设四旋翼需要克服重力和张力
            gravity_force = self.quad_mass * self.g
            
            # 绳子张力在四旋翼上的作用力
            cable_angle = self.cable.state[:, i, 0]
            d_i = torch.stack([
                torch.cos(cable_angle),
                torch.sin(cable_angle),
                torch.zeros_like(cable_angle)
            ], dim=1)
            
            t_i = tensions[:, i]
            tension_force_magnitude = t_i
            
            # 总推力需求（简化为重力 + 张力的垂直分量）
            required_thrust = gravity_force + torch.abs(d_i[:, 2]) * tension_force_magnitude
            
            # 推力约束违反
            thrust_violation = torch.clamp(required_thrust - self.f_max, min=0)
            penalty += self.constraint_penalty_weight * thrust_violation
        
        return -penalty  # 返回负惩罚

    def _check_done(self):
        """
        检查终止条件
        """
        # 1. 时间步数达到上限
        if self.step_counter >= self.total_timesteps:
            return True
            
        # 2. 负载高度过低
        current_pos = self.payload.state[:, 0:3]
        if torch.any(current_pos[:, 2] < self.min_height):
            return True
            
        # 3. 跟踪误差过大
        target_pos = load_trajectory_point(self.current_point_index).unsqueeze(0).expand(self.batch_size, 3)
        tracking_error = torch.norm(current_pos - target_pos, dim=1)
        if torch.any(tracking_error > self.max_tracking_error):
            return True
        
        # 4. 四旋翼碰撞检测
        quad_positions = self.compute_quad_positions()
        
        # 四旋翼与障碍物碰撞
        for i in range(self.n_cables):
            for obs_pos in self.obstacle_pos:
                obs_pos_3d = torch.cat([obs_pos, current_pos[:, 2:3]], dim=0)
                dist_to_obs = torch.norm(quad_positions[:, i] - obs_pos_3d.unsqueeze(0), dim=1)
                if torch.any(dist_to_obs < self.obstacle_r + self.collision_tolerance):
                    return True
        
        # 四旋翼之间碰撞
        for i in range(self.n_cables):
            for j in range(i+1, self.n_cables):
                dist_ij = torch.norm(quad_positions[:, i] - quad_positions[:, j], dim=1)
                if torch.any(dist_ij < self.collision_tolerance):
                    return True
                    
        # 5. 轨迹完成
        if self.current_point_index >= self.trajectory_length - 1:
            return True
            
        return False

    def _get_obs(self):
        B = self.batch_size
        
        # r_g (3维) - 负载在世界坐标系中的位置
        r_g = self.payload.state[:, 0:3]  # (B,3)
        
        # 障碍物信息 (每个障碍物3维: x, y, r)
        obstacle_info = []
        for pos in self.obstacle_pos:
            obs_info = torch.cat([
                pos, 
                torch.tensor([self.obstacle_r], device=self.device)
            ]).unsqueeze(0).expand(B, 3)
            obstacle_info.append(obs_info)
        obstacle_info = torch.cat(obstacle_info, dim=1)  # (B, n_obstacles*3)
        
        # 下一时刻轨迹点 (3维)
        next_point = load_trajectory_point(self.current_point_index).unsqueeze(0).expand(B, 3)

        # 组合观测
        obs = torch.cat([r_g, obstacle_info, next_point], dim=1)
        return obs[0].cpu().numpy()

    def render(self, mode="human"):
        """
        可视化环境状态
        """
        if mode == "human":
            # 获取当前状态信息
            payload_pos = self.payload.state[0, 0:3].cpu().numpy()
            quad_positions = self.compute_quad_positions()[0].cpu().numpy()
            target_pos = load_trajectory_point(self.current_point_index).cpu().numpy()
            
            print(f"Step: {self.step_counter}")
            print(f"Payload Position: {payload_pos}")
            print(f"Quad Positions: {quad_positions}")
            print(f"Target Position: {target_pos}")