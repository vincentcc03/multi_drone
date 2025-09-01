import numpy as np
from src.utils.read_yaml import load_config
import os
import torch

def load_npy():
# 加载 .npy 文件
    loaded_data = np.load('document/Reference_traj_4/coeffx2.npy')

    print(loaded_data)
    print(f"形状: {loaded_data.shape}")
    print(f"数据类型: {loaded_data.dtype}")

def load_trajectory_point(point_index):
    config = load_config("env_config.yaml")
    traj_number = config.get("trajectory_numbers")
    
    base_path = config.get("base_path", "document/Reference_traj_4")
    coeffx_pattern = config.get("coeffx_pattern", "coeffx{}.npy")
    
    # 构建文件路径
    x_file = os.path.join(base_path, coeffx_pattern.format(traj_number))
    y_file = os.path.join(base_path, coeffx_pattern.replace('x', 'y').format(traj_number))
    z_file = os.path.join(base_path, coeffx_pattern.replace('x', 'z').format(traj_number))
    
    # 加载数据
    x = torch.from_numpy(np.load(x_file)).float()[point_index]
    y = torch.from_numpy(np.load(y_file)).float()[point_index]
    z = torch.from_numpy(np.load(z_file)).float()[point_index]
    
    return torch.tensor([x, y, z])