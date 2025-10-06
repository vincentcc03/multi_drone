import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os
from datetime import datetime

# 控制print输出的开关
ENABLE_PRINT = False  # 设置为False即可关闭所有print

# 重定义print函数
if not ENABLE_PRINT:
    def print(*args, **kwargs):
        pass

class DroneVisualization:
    def __init__(self, save_dir="results/ppo"):
        self.save_dir = save_dir
        self.fig_count = 0
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_scene(self, payload_pos, drone_positions, ref_trajectory, 
                       obstacle_positions, obstacle_radius, payload_radius, 
                       drone_radius, current_step, rope_length=1.0):
        """
        可视化整个场景
        
        Args:
            payload_pos: 负载位置 (3,) numpy array
            drone_positions: 无人机位置 (N, 3) numpy array
            ref_trajectory: 参考轨迹 (T, 3) numpy array
            obstacle_positions: 障碍物位置 (M, 2) numpy array
            obstacle_radius: 障碍物半径
            payload_radius: 负载半径
            drone_radius: 无人机半径
            current_step: 当前步数
            rope_length: 绳索长度
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 调试信息
        print(f"Payload pos: {payload_pos}")
        print(f"Drone positions shape: {drone_positions.shape if drone_positions is not None else None}")
        
        # 绘制参考轨迹
        if len(ref_trajectory) > 0:
            ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
                   'b--', linewidth=2, alpha=0.6, label='Reference Trajectory')
        
        # 绘制负载（红色大点）
        if payload_pos is not None:
            ax.scatter(payload_pos[0], payload_pos[1], payload_pos[2], 
                      c='red', s=200, alpha=0.8, label='Payload', marker='o')
            print(f"Drawing payload at {payload_pos}")
        
        # 绘制无人机（绿色点）并连接到负载
        if drone_positions is not None and payload_pos is not None:
            for i, drone_pos in enumerate(drone_positions):
                # 绘制无人机点
                ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], 
                          c='green', s=100, alpha=0.8, 
                          label='Drones' if i == 0 else "", marker='^')
                
                # 绘制连接线（绳索）
                ax.plot([payload_pos[0], drone_pos[0]], 
                       [payload_pos[1], drone_pos[1]], 
                       [payload_pos[2], drone_pos[2]], 
                       'k-', linewidth=2, alpha=0.6)
                
                # 添加无人机编号标注
                ax.text(drone_pos[0], drone_pos[1], drone_pos[2] + 0.2, 
                       f'D{i+1}', fontsize=10, ha='center')
                
                print(f"Drawing drone {i} at {drone_pos}")
        
        # 绘制障碍物（垂直线）
        if len(obstacle_positions) > 0:
            for i, obs_pos in enumerate(obstacle_positions):
                # 从地面到一定高度画垂直线
                obstacle_height = 3.0  # 障碍物高度
                ax.plot([obs_pos[0], obs_pos[0]], 
                       [obs_pos[1], obs_pos[1]], 
                       [0, obstacle_height], 
                       'gray', linewidth=8, alpha=0.8, 
                       label='Obstacles' if i == 0 else "")
                
                # 在障碍物顶部添加圆圈表示范围
                theta = np.linspace(0, 2*np.pi, 20)
                circle_x = obs_pos[0] + obstacle_radius * np.cos(theta)
                circle_y = obs_pos[1] + obstacle_radius * np.sin(theta)
                circle_z = np.full_like(circle_x, obstacle_height)
                ax.plot(circle_x, circle_y, circle_z, 'gray', linewidth=2, alpha=0.5)
                
                # 添加障碍物标注
                ax.text(obs_pos[0], obs_pos[1], obstacle_height + 0.2, 
                       f'Obs{i+1}', fontsize=10, ha='center')
        
        # 绘制地面网格
        self._draw_ground_grid(ax)
        
        # 动态设置坐标轴范围
        all_positions = []
        
        # 收集所有位置点
        if payload_pos is not None:
            all_positions.append(payload_pos)
        
        if drone_positions is not None:
            all_positions.extend(drone_positions)
        
        if len(ref_trajectory) > 0:
            all_positions.extend(ref_trajectory[::10])  # 采样参考轨迹点
        
        if len(obstacle_positions) > 0:
            for obs_pos in obstacle_positions:
                all_positions.append([obs_pos[0], obs_pos[1], 1.5])
        
        # 计算合适的坐标轴范围
        if all_positions:
            all_positions = np.array(all_positions)
            x_min, x_max = all_positions[:, 0].min() - 2, all_positions[:, 0].max() + 2
            y_min, y_max = all_positions[:, 1].min() - 2, all_positions[:, 1].max() + 2
            z_min, z_max = 0, max(all_positions[:, 2].max() + 2, 4)
            
            # 确保坐标轴范围不会太小
            x_range = max(x_max - x_min, 6)
            y_range = max(y_max - y_min, 6)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
            ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
            ax.set_zlim([z_min, z_max])
            
            print(f"Axis ranges: X[{x_center - x_range/2:.2f}, {x_center + x_range/2:.2f}]")
            print(f"             Y[{y_center - y_range/2:.2f}, {y_center + y_range/2:.2f}]")
            print(f"             Z[{z_min:.2f}, {z_max:.2f}]")
        else:
            # 默认范围
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_zlim([0, 4])
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='upper left')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 设置标题
        timestamp = datetime.now().strftime("%H:%M:%S")
        ax.set_title(f'Multi-Drone Payload Transport System\nStep: {current_step}, Time: {timestamp}')
        
        # 保存图片
        filename = f"visualization_step_{current_step:06d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {filepath}")
        self.fig_count += 1
    
    def _draw_ground_grid(self, ax):
        """绘制地面网格"""
        # 获取当前坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # 创建网格
        x_grid = np.arange(xlim[0], xlim[1], 1.0)
        y_grid = np.arange(ylim[0], ylim[1], 1.0)
        
        # 绘制网格线
        for x in x_grid:
            ax.plot([x, x], [ylim[0], ylim[1]], [0, 0], 'lightgray', alpha=0.3, linewidth=0.5)
        
        for y in y_grid:
            ax.plot([xlim[0], xlim[1]], [y, y], [0, 0], 'lightgray', alpha=0.3, linewidth=0.5)
    
    def plot_trajectory_comparison(self, actual_traj, ref_traj, save_name="trajectory_comparison"):
        """
        绘制实际轨迹与参考轨迹的对比图
        
        Args:
            actual_traj: 实际轨迹 (T, 3) numpy array
            ref_traj: 参考轨迹 (T, 3) numpy array
            save_name: 保存文件名
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 3D轨迹对比
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 
                'b-', linewidth=2, label='Reference', alpha=0.8)
        ax1.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], 
                'r-', linewidth=2, label='Actual', alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('3D Trajectory Comparison')
        
        # XY平面轨迹
        ax2 = fig.add_subplot(132)
        ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=2, label='Reference', alpha=0.8)
        ax2.plot(actual_traj[:, 0], actual_traj[:, 1], 'r-', linewidth=2, label='Actual', alpha=0.8)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.set_title('XY Plane Trajectory')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 高度随时间变化
        ax3 = fig.add_subplot(133)
        time_steps = np.arange(len(ref_traj)) * 0.01  # 假设dt=0.01
        ax3.plot(time_steps, ref_traj[:, 2], 'b-', linewidth=2, label='Reference', alpha=0.8)
        ax3.plot(time_steps, actual_traj[:, 2], 'r-', linewidth=2, label='Actual', alpha=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Z (m)')
        ax3.legend()
        ax3.set_title('Altitude vs Time')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory comparison saved to: {filepath}")
    
    def create_animation_frames(self):
        """可选：创建动画帧（如果需要制作gif动画）"""
        # 这里可以添加将所有图片合成gif的代码
        pass