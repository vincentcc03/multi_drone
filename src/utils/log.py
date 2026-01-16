import os
from datetime import datetime
from src.utils.read_yaml import load_config
class TrainingLogger:
    def __init__(self, log_dir="log", log_filename=None):
        self.config = load_config("ppo_config.yaml")
        self.env_config = load_config("env_config.yaml")
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志文件夹路径
            log_filename: 自定义日志文件名，如果为None则自动生成带时间戳的文件名
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"training_log_{timestamp}.txt"
        
        self.log_file = os.path.join(self.log_dir, log_filename)
        
        # 初始化日志文件
        self._init_log_file()
    
    def _init_log_file(self):
        """初始化日志文件头部"""
        num_actors = self.config["params"]["config"].get("num_actors")  # 并行环境数量
        horizon_length = self.config["params"]["config"].get("horizon_length")  # 每次收集的 rollout 长度
        minibatch_size = self.config["params"]["config"].get("minibatch_size")  # 每个小批量大小
        mini_epochs = self.config["params"]["config"].get("mini_epochs")  # 每次更新的迭代次数
        learning_rate = self.config["params"]["config"].get("learning_rate")  # 初始学习率
        e_clip = self.config["params"]["config"].get("e_clip")  # PPO 的 clip ε
        gamma = self.config["params"]["config"].get("gamma")  # 折扣因子 γ
        tau = self.config["params"]["config"].get("tau")  # GAE(λ) 中的 λ
        max_epochs = self.config["params"]["config"].get("max_epochs")  # 最大训练轮数
        pos_error_threshold = self.env_config.get("pos_error_threshold", 0.15)  # 位置误差阈值
        pos_error_count = self.env_config.get("pos_error_count", 5)  # 连续位置误差超过阈值的步数
        progress_reward = self.env_config.get("progress_reward")  # 进度奖励
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Num Actors: {num_actors}\n")
            f.write(f"Horizon Length: {horizon_length}\n")
            f.write(f"Minibatch Size: {minibatch_size}\n")
            f.write(f"Mini Epochs: {mini_epochs}\n") 
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"E-Clip: {e_clip}\n")
            f.write(f"Gamma: {gamma}\n")
            f.write(f"Tau: {tau}\n")
            f.write(f"Max Epochs: {max_epochs}\n")
            f.write(f"Position Error Threshold: {pos_error_threshold}\n")
            f.write(f"Position Error Count: {pos_error_count}\n")
            f.write(f"Progress Reward: {progress_reward}\n")
            f.write("-" * 50 + "\n")
            f.write("max_progress,env1_Reward,Position,Velocity,Quaternion,Angular_Velocity,drone_payload,drone_drone,Action_Smoothness,Trajectory_Progress_Reward\n")

    def log_step(self, traj_progress,reward1, pos, vel, quat, omega, drone_payload, drone_drone, action_smoothness, traj_progress_reward):
        """
        记录单步训练数据
        
        Args:
            traj_progress: 当前轨迹进度
            pos: 位置奖励
            vel: 速度奖励
            quat: 四元数奖励
            omega: 角速度奖励
            drone_payload: 无人机负载奖励
            drone_drone: 无人机间奖励
            action_smoothness: 动作平滑度奖励
            traj_progress_reward: 轨迹进度奖励
        """
        timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{traj_progress},{reward1:.6f},{pos:.6f},{vel:.6f},{quat:.6f},{omega:.6f},{drone_payload:.6f},{drone_drone:.6f},{action_smoothness:.6f},{traj_progress_reward:.6f}\n")

    def log_episode(self, episode, max_reward, episode_reward=None):
        """
        记录回合数据
        
        Args:
            episode: 回合数
            max_reward: 最大奖励
            episode_reward: 回合奖励（可选）
        """
        timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            if episode_reward is not None:
                f.write(f"Episode {episode},{max_reward:.6f},{episode_reward:.6f},{timestamp}\n")
            else:
                f.write(f"Episode {episode},{max_reward:.6f},{timestamp}\n")
    
    def get_log_file_path(self):
        """返回日志文件路径"""
        return self.log_file