import os
import time
import threading
from datetime import datetime

class FolderMonitor:
    def __init__(self, folder_path, threshold=100, check_interval=60):
        """
        持续监控文件夹并清理图片
        
        Args:
            folder_path: 要监控的文件夹路径
            threshold: 图片数量阈值
            check_interval: 检查间隔（秒）
        """
        self.folder_path = folder_path
        self.threshold = threshold
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # 定义常见图片扩展名
        self.image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.svg')
        
    def get_image_files(self):
        """获取文件夹中的所有图片文件"""
        if not os.path.exists(self.folder_path):
            return []
            
        try:
            files = os.listdir(self.folder_path)
            images = [f for f in files
                     if f.lower().endswith(self.image_exts) 
                     and os.path.isfile(os.path.join(self.folder_path, f))]
            return images
        except Exception as e:
            print(f"读取文件夹 {self.folder_path} 时出错：{e}")
            return []
    
    def clear_images(self, images):
        """删除指定的图片文件"""
        deleted_count = 0
        for filename in images:
            filepath = os.path.join(self.folder_path, filename)
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"删除 {filename} 时出错：{e}")
        return deleted_count
    
    def check_and_clean(self):
        """检查并清理文件夹"""
        images = self.get_image_files()
        num_images = len(images)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] 检查文件夹 {self.folder_path}，找到 {num_images} 张图片")
        
        if num_images > self.threshold:
            print(f"图片数量超过阈值 {self.threshold}，开始清理...")
            deleted_count = self.clear_images(images)
            print(f"成功删除 {deleted_count} 张图片")
        else:
            print(f"图片数量未超过阈值，无需清理")
    
    def monitor_loop(self):
        """监控循环"""
        print(f"开始监控文件夹: {self.folder_path}")
        print(f"阈值: {self.threshold} 张图片")
        print(f"检查间隔: {self.check_interval} 秒")
        print("按 Ctrl+C 停止监控\n")
        
        while self.is_running:
            try:
                self.check_and_clean()
                print(f"等待 {self.check_interval} 秒后进行下次检查...\n")
                
                # 使用短间隔检查停止标志，实现更快响应
                for _ in range(self.check_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n收到停止信号，正在退出...")
                break
            except Exception as e:
                print(f"监控过程中出错：{e}")
                time.sleep(5)  # 出错后短暂等待
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            print("监控已在运行中")
            return
              
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        print("监控已停止")

def clear_folder_if_too_many_images(folder_path, threshold=100):
    """一次性清理函数（向后兼容）"""
    monitor = FolderMonitor(folder_path, threshold)
    monitor.check_and_clean()

def start_continuous_monitoring(folders=None,  =100, check_interval=60):
    """启动多文件夹持续监控"""
    if folders is None:
        folders = ["results/payload", "results/ppo"]
    
    monitors = []
    
    try:
        # 为每个文件夹创建监控器
        for folder in folders:
            if os.path.exists(folder):
                monitor = FolderMonitor(folder, threshold, check_interval)
                monitor.start_monitoring()
                monitors.append(monitor)
                time.sleep(1)  # 错开启动时间
            else:
                print(f"警告：文件夹 {folder} 不存在，跳过监控")
        
        if not monitors:
            print("没有找到可监控的文件夹")
            return
        
        # 保持主线程运行
        print("所有监控器已启动，按 Ctrl+C 停止所有监控")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止所有监控器...")
        for monitor in monitors:
            monitor.stop_monitoring()
        print("所有监控已停止")

if __name__ == "__main__":
    # 方式1：单文件夹一次性检查
    # folder = "results/payload"
    # clear_folder_if_too_many_images(folder, threshold=50)
    
    # 方式2：单文件夹持续监控
    # monitor = FolderMonitor("results/payload", threshold=50, check_interval=30)
    # monitor.start_monitoring()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     monitor.stop_monitoring()
    
    # 方式3：多文件夹持续监控
    folders_to_monitor = [
        "results/payload",
    ]
    start_continuous_monitoring(
        folders=folders_to_monitor,
        threshold=1000,
        check_interval=60  # 每60秒检查一次
    )