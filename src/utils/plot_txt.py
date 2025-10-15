import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from datetime import datetime

# ====== 1. 读取日志文件 ======
log_path = "log/training_log_20251012_011847.txt"

# 解析数据
data = []
with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

    # 找到数据开始的位置
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Current_Step"):
            start_idx = i + 1
            break

    # 提取数据
    for line in lines[start_idx:]:
        parts = line.strip().split(",")
        if len(parts) == 3:
            try:
                current_step = int(parts[0])
                max_reward = float(parts[1])
                timestamp = parts[2]
                data.append([current_step, max_reward, timestamp])
            except ValueError:
                continue

# ====== 2. 转为 DataFrame ======
df = pd.DataFrame(data, columns=["Current_Step", "Max_Reward", "Timestamp"])

# ====== 3. 绘图 ======
plt.figure(figsize=(10, 6))

# 曲线 1：Reward
plt.plot(df["Max_Reward"], label="Max Reward", color="tab:blue", linewidth=2)

# 曲线 2：Current Step（归一化以便对比）
plt.plot(df["Current_Step"] / df["Current_Step"].max(), label="Current Step (normalized)", color="tab:orange", linestyle="--")

plt.title("Training Progress")
plt.xlabel("Data Point Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ====== 4. 保存图片 ======
# 创建保存目录
os.makedirs("results", exist_ok=True)

# 生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"results/training_progress_{timestamp}.png"

# 保存图片
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"训练进度图已保存至: {save_path}")

# 显示图片
plt.show()