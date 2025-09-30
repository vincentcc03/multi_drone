import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from src.envs.env import RLGamesEnv
from src.envs.RLGamesEnvWrapper import RLGamesEnvWrapper
import os

# -------------------------------
# 1️⃣ 环境构造函数（与训练脚本相同）
# -------------------------------
def create_env(config_name=None, num_envs=1, **kwargs):
    env = RLGamesEnv(
        config_name="env_config.yaml",
        traj_name="traj_config.yaml", 
        num_envs=num_envs,
        device="cuda"
    )
    return RLGamesEnvWrapper(env)

# -------------------------------
# 2️⃣ 注册 vecenv（与训练脚本相同）
# -------------------------------
vecenv.register(
    "RLGPUEnv",
    lambda config_name, num_actors, **kwargs: create_env(num_envs=num_actors, **kwargs)
)

# -------------------------------
# 3️⃣ 注册环境到 rl-games 配置（与训练脚本相同）
# -------------------------------
env_configurations.register(
    "rlgpu",
    {
        "vecenv_type": "RLGPUEnv", 
        "env_creator": lambda **kwargs: create_env(**kwargs)
    }
)

def create_test_config(base_config_path, checkpoint_path):
    """基于训练配置创建测试配置"""
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 修改为测试模式
    config['params']['config']['is_train'] = False
    config['params']['config']['num_actors'] = 1  # 测试时只用1个环境
    config['params']['config']['player'] = True   # 添加 player 模式
    config['params']['load_checkpoint'] = True
    config['params']['load_path'] = checkpoint_path
    
    # 确保路径格式正确
    if not os.path.isabs(checkpoint_path):
        config['params']['load_path'] = os.path.abspath(checkpoint_path)
    
    return config

def test_model(config, num_episodes=10):
    """测试训练好的模型"""
    
    # 创建 runner 并加载配置
    runner = Runner()
    runner.load(config)
    runner.reset()
    
    # 设置为评估模式 - 修正访问方式
    if hasattr(runner, 'agent') and hasattr(runner.agent, 'model'):
        runner.agent.model.eval()
    elif hasattr(runner, 'algo') and hasattr(runner.algo, 'model'):
        runner.algo.model.eval()
    
    results = []
    
    print(f"开始测试，共 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        # 修正环境重置方式
        obs = runner.env_reset()  # 使用 runner 的重置方法
        episode_reward = 0
        episode_length = 0
        done = False
        
        positions = []  # 记录负载位置轨迹
        target_positions = []  # 记录目标位置
        rewards = []  # 记录每步奖励
        
        while not done and episode_length < 1000:
            with torch.no_grad():
                # 修正动作获取方式
                if hasattr(runner, 'agent'):
                    action = runner.agent.get_action(obs, is_deterministic=True)
                elif hasattr(runner, 'algo'):
                    action = runner.algo.get_action(obs, is_deterministic=True)
                else:
                    # 备用方案：直接使用模型
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda')
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action = runner.algo.model.a2c_network(obs_tensor)['mus'].cpu().numpy()
            
            # 执行动作 - 修正环境步进方式
            obs, reward, done, info = runner.env_step(action)
            
            # 处理奖励格式
            if isinstance(reward, (list, np.ndarray)):
                reward_val = reward[0] if len(reward) > 0 else 0
            elif isinstance(reward, torch.Tensor):
                reward_val = reward.item() if reward.numel() == 1 else reward[0].item()
            else:
                reward_val = reward
                
            episode_reward += reward_val
            episode_length += 1
            rewards.append(reward_val)
            
            # 处理 done 格式
            if isinstance(done, (list, np.ndarray)):
                done = done[0] if len(done) > 0 else False
            elif isinstance(done, torch.Tensor):
                done = done.item() if done.numel() == 1 else done[0].item()
            
            # 记录位置信息
            if len(info) > 0 and isinstance(info[0], dict):
                if 'current_position' in info[0]:
                    positions.append(info[0]['current_position'].cpu().numpy())
                if 'target_position' in info[0]:
                    target_positions.append(info[0]['target_position'].cpu().numpy())
        
        final_error = info[0].get('pos_error', 0) if len(info) > 0 and isinstance(info[0], dict) else 0
        
        results.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'positions': np.array(positions) if positions else np.array([]),
            'targets': np.array(target_positions) if target_positions else np.array([]),
            'rewards': np.array(rewards),
            'final_error': final_error
        })
        
        print(f"回合 {episode+1}: 总奖励={episode_reward:.2f}, 长度={episode_length}, 最终误差={final_error:.3f}")
    
    return results

def plot_results(results):
    """可视化测试结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 奖励分布
    rewards = [r['reward'] for r in results]
    axes[0,0].bar(range(1, len(rewards)+1), rewards, color='skyblue', alpha=0.7)
    axes[0,0].set_title('每回合总奖励')
    axes[0,0].set_xlabel('回合')
    axes[0,0].set_ylabel('总奖励')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 回合长度
    lengths = [r['length'] for r in results]
    axes[0,1].bar(range(1, len(lengths)+1), lengths, color='lightgreen', alpha=0.7)
    axes[0,1].set_title('每回合长度')
    axes[0,1].set_xlabel('回合')
    axes[0,1].set_ylabel('步数')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 最终误差
    errors = [r['final_error'] for r in results]
    axes[0,2].bar(range(1, len(errors)+1), errors, color='salmon', alpha=0.7)
    axes[0,2].set_title('最终位置误差')
    axes[0,2].set_xlabel('回合')
    axes[0,2].set_ylabel('误差 (m)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 轨迹跟踪 (选择第一个回合)
    if len(results) > 0 and len(results[0]['positions']) > 0:
        pos = results[0]['positions']
        targets = results[0]['targets']
        
        axes[1,0].plot(pos[:, 0], pos[:, 1], 'b-', label='实际轨迹', linewidth=2, alpha=0.8)
        axes[1,0].plot(targets[:, 0], targets[:, 1], 'r--', label='目标轨迹', linewidth=2, alpha=0.8)
        axes[1,0].set_title('轨迹跟踪效果 (回合 1)')
        axes[1,0].set_xlabel('X (m)')
        axes[1,0].set_ylabel('Y (m)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axis('equal')
    
    # 5. 奖励随时间变化 (选择第一个回合)
    if len(results) > 0 and len(results[0]['rewards']) > 0:
        rewards_over_time = results[0]['rewards']
        axes[1,1].plot(rewards_over_time, 'g-', alpha=0.7)
        axes[1,1].set_title('每步奖励变化 (回合 1)')
        axes[1,1].set_xlabel('步数')
        axes[1,1].set_ylabel('奖励')
        axes[1,1].grid(True, alpha=0.3)
    
    # 6. 统计分布
    axes[1,2].hist(rewards, bins=min(10, len(rewards)), alpha=0.7, color='purple')
    axes[1,2].set_title('奖励分布')
    axes[1,2].set_xlabel('总奖励')
    axes[1,2].set_ylabel('频次')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("📊 测试结果统计")
    print("="*50)
    print(f"🎯 平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"⏱️  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"📍 平均最终误差: {np.mean(errors):.3f} ± {np.std(errors):.3f}")
    print(f"✅ 成功率 (误差 < 0.1m): {np.mean(np.array(errors) < 0.1)*100:.1f}%")
    print(f"🏆 最佳奖励: {np.max(rewards):.2f}")
    print(f"💯 最小误差: {np.min(errors):.3f}")

def main():
    # 配置文件路径
    base_config_path = "src/config/ppo_config.yaml"
    
    # 检查最新的检查点文件
    checkpoint_dir = "runs"
    checkpoint_files = []
    
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith("last_lift_") and file.endswith(".pth"):
                checkpoint_files.append(os.path.join(root, file))
    
    if not checkpoint_files:
        print("❌ 未找到检查点文件！请先运行训练脚本。")
        return
    
    # 选择最新的检查点
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"🔄 使用检查点: {latest_checkpoint}")
    
    # 创建测试配置
    test_config = create_test_config(base_config_path, latest_checkpoint)
    
    # 运行测试
    print("🚀 开始模型测试...")
    results = test_model(test_config, num_episodes=5)
    
    # 可视化结果
    print("\n📈 生成测试报告...")
    plot_results(results)
    
    print("\n✅ 测试完成！结果已保存为 test_results.png")

if __name__ == "__main__":
    main()