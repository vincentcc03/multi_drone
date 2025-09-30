import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
import os
from datetime import datetime

class DroneVisualization:
    def __init__(self, save_dir="results/ppo"):
        self.save_dir = save_dir
        self.fig_count = 0
        os.makedirs(save_dir, exist_ok=True)
        
    def create_cube_vertices(self, center, size):
        """创建立方体的顶点"""
        cx, cy, cz = center
        s = size / 2
        
        vertices = np.array([
            [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
            [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
            [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
            [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s]
        ])
        return vertices
    
    def create_cube_faces(self, vertices):
        """创建立方体的面"""
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # left
        ]
        return faces
    
    def plot_cube(self, ax, center, size, color, alpha=0.7, label=None):
        """在3D坐标轴上绘制立方体"""
        vertices = self.create_cube_vertices(center, size)
        faces = self.create_cube_faces(vertices)
        
        cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(cube)  # 修改这里：add_3d_collection -> add_collection3d
        
        if label:
            ax.text(center[0], center[1], center[2] + size/2, label, 
                   fontsize=8, ha='center')
    
    def visualize_scene(self, payload_pos, drone_positions, ref_trajectory, 
                       obstacle_positions, obstacle_radius, payload_radius, 
                       drone_radius, current_step):
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
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 调试信息
        print(f"Payload pos: {payload_pos}")
        print(f"Drone positions shape: {drone_positions.shape if drone_positions is not None else None}")
        print(f"Payload radius: {payload_radius}, Drone radius: {drone_radius}")
        
        # 绘制参考轨迹
        if len(ref_trajectory) > 0:
            ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
                   'b--', linewidth=2, alpha=0.6, label='Reference Trajectory')
        
        # 绘制负载（红色立方体） - 增大尺寸
        if payload_pos is not None:
            cube_size = max(payload_radius * 4, 0.5)  # 确保立方体足够大
            self.plot_cube(ax, payload_pos, cube_size, 'red', 
                          alpha=0.8, label='Payload')
            print(f"Drawing payload at {payload_pos} with size {cube_size}")
        
        # 绘制无人机（绿色立方体） - 增大尺寸
        if drone_positions is not None:
            cube_size = max(drone_radius * 4, 0.3)  # 确保立方体足够大
            for i, drone_pos in enumerate(drone_positions):
                self.plot_cube(ax, drone_pos, cube_size, 'green', 
                              alpha=0.7, label=f'Drone {i+1}' if i == 0 else None)
                print(f"Drawing drone {i} at {drone_pos} with size {cube_size}")
        
        # 绘制障碍物（灰色立方体）
        if len(obstacle_positions) > 0:
            for i, obs_pos in enumerate(obstacle_positions):
                obs_pos_3d = np.array([obs_pos[0], obs_pos[1], 0.5])
                cube_size = max(obstacle_radius * 4, 0.4)
                self.plot_cube(ax, obs_pos_3d, cube_size, 'gray', 
                              alpha=0.6, label='Obstacle' if i == 0 else None)
        
        # 动态设置坐标轴范围
        all_positions = []
        
        # 收集所有位置点
        if payload_pos is not None:
            all_positions.append(payload_pos)
        
        if drone_positions is not None:
            all_positions.extend(drone_positions)
        
        if len(ref_trajectory) > 0:
            all_positions.extend(ref_trajectory)
        
        if len(obstacle_positions) > 0:
            for obs_pos in obstacle_positions:
                all_positions.append([obs_pos[0], obs_pos[1], 0.5])
        
        # 计算合适的坐标轴范围
        if all_positions:
            all_positions = np.array(all_positions)
            x_min, x_max = all_positions[:, 0].min() - 2, all_positions[:, 0].max() + 2
            y_min, y_max = all_positions[:, 1].min() - 2, all_positions[:, 1].max() + 2
            z_min, z_max = all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1
            
            # 确保坐标轴范围不会太小
            x_range = max(x_max - x_min, 4)
            y_range = max(y_max - y_min, 4)
            z_range = max(z_max - z_min, 2)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            
            ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
            ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
            ax.set_zlim([z_center - z_range/2, z_center + z_range/2])
            
            print(f"Axis ranges: X[{x_center - x_range/2:.2f}, {x_center + x_range/2:.2f}]")
            print(f"             Y[{y_center - y_range/2:.2f}, {y_center + y_range/2:.2f}]")
            print(f"             Z[{z_center - z_range/2:.2f}, {z_center + z_range/2:.2f}]")
        else:
            # 默认范围
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_zlim([0, 4])
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        # 设置标题
        timestamp = datetime.now().strftime("%H:%M:%S")
        ax.set_title(f'Multi-Drone System Visualization\nStep: {current_step}, Time: {timestamp}')
        
        # 保存图片
        filename = f"visualization_step_{current_step:06d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {filepath}")
        self.fig_count += 1
        
    def create_animation_frames(self):
        """可选：创建动画帧（如果需要制作gif动画）"""
        # 这里可以添加将所有图片合成gif的代码
        pass