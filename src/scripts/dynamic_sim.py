import torch
from src.envs.dynamics.payload_dynamics import PayloadDynamicsSimBatch
from src.utils.plot import plot_trajectories, plot_trajectories_grid
from src.utils.read_yaml import load_config

def simulate_batch(sim_class, input_force_torque_seq, steps):
    sims = sim_class()
    
    trajs = []
    for i in range(steps):
        trajs.append(sims.state.clone())
        sims.rk4_step(input_force_torque_seq[i])
    return torch.stack(trajs)  # (steps, batch, 13)
def run_simulation(config_path="env_config.yaml"):
    """
    运行批量仿真的主函数
    
    Args:
        config_path (str): 配置文件路径，默认为 "env_config.yaml"
    
    Returns:
        torch.Tensor: 仿真轨迹数据，形状为 (steps, batch_size, 13)
    """
    # 加载配置
    config = load_config(config_path)
    steps = config["steps"]
    batch_size = config["batch_size"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # 创建输入力/力矩序列
    input_force_torque_seq = torch.zeros(steps, batch_size, 6, device=device)
    for b in range(batch_size):
        input_force_torque_seq[:, b, 2] = 40  # z方向力不同

    # 运行仿真
    trajs = simulate_batch(
        PayloadDynamicsSimBatch,
        input_force_torque_seq=input_force_torque_seq,
        steps=steps,
    )
    
    # 可视化结果
    plot_trajectories_grid(trajs)
    plot_trajectories(trajs)
    
    print(f"仿真完成！设备: {trajs.device}")
    print(f"轨迹形状: {trajs.shape}")
    
    return trajs


if __name__ == "__main__":
    # 使用默认配置运行
    trajectories = run_simulation()
    
    # 或者使用自定义配置
    # trajectories = run_simulation("custom_config.yaml")
    
    # 可以在这里对 trajectories 进行进一步处理
    # 例如：分析结果、保存数据等