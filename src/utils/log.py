import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="log", log_filename=None):
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
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            f.write("Current_Step,Max_Reward,Timestamp\n")
    
    def log_step(self, current_step, max_reward):
        """
        记录单步训练数据
        
        Args:
            current_step: 当前步数
            max_reward: 最大奖励
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{current_step},{max_reward:.6f},{timestamp}\n")
    
    def log_episode(self, episode, max_reward, episode_reward=None):
        """
        记录回合数据
        
        Args:
            episode: 回合数
            max_reward: 最大奖励
            episode_reward: 回合奖励（可选）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            if episode_reward is not None:
                f.write(f"Episode {episode},{max_reward:.6f},{episode_reward:.6f},{timestamp}\n")
            else:
                f.write(f"Episode {episode},{max_reward:.6f},{timestamp}\n")
    
    def get_log_file_path(self):
        """返回日志文件路径"""
        return self.log_file