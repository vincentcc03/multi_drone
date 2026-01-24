import torch


def smooth_trajectory(trajectory, num_points_between=9):
    """
    参考轨迹插点平滑函数
    
    将原始轨迹（101个点）通过在每两个点之间插入9个点，得到1001个点的平滑轨迹
    
    Args:
        trajectory: torch.Tensor, 形状 (101, 13)，原始参考轨迹
        num_points_between: int, 每两个点之间插入的点数（默认为9）
    
    Returns:
        torch.Tensor, 形状 (1001, 13)，平滑后的轨迹
    
    说明:
        - 原始轨迹：101个点
        - 每两个相邻点之间插入9个点
        - 最终轨迹：(101-1) * (9+1) + 1 = 100 * 10 + 1 = 1001个点
        - 使用线性插值实现平滑
    """
    if trajectory.dim() == 1:
        trajectory = trajectory.unsqueeze(0)
    
    num_original_points = trajectory.shape[0]
    dim = trajectory.shape[1]
    device = trajectory.device
    
    # 总点数 = (原始点数 - 1) * (插入点数 + 1) + 1
    total_points = (num_original_points - 1) * (num_points_between + 1) + 1
    
    # 创建平滑轨迹张量
    smooth_traj = torch.zeros(total_points, dim, device=device, dtype=trajectory.dtype)
    
    # 线性插值
    idx = 0
    for i in range(num_original_points - 1):
        # 当前点和下一个点
        p_start = trajectory[i]
        p_end = trajectory[i + 1]
        
        # 在这两个点之间插入 num_points_between 个点（包括起点，不包括终点）
        for j in range(num_points_between + 1):
            alpha = j / (num_points_between + 1)  # 插值系数 0 到 1 之间
            smooth_traj[idx] = (1 - alpha) * p_start + alpha * p_end
            idx += 1
    
    # 最后一个点
    smooth_traj[-1] = trajectory[-1]
    
    return smooth_traj
