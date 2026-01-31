#!/usr/bin/env python
"""
读取训练日志文件并绘制 reward 图像
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_reward_from_log(log_file):
    """
    从日志文件读取数据并绘制所有列的图像
    
    Args:
        log_file: 日志文件路径
    """
    # 读取 CSV 文件（跳过前面的配置行）
    try:
        df = pd.read_csv(log_file, skiprows=15)  # 跳过头部配置行
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 获取所有列
    columns_to_plot = list(df.columns)
    
    if not columns_to_plot:
        print(f"没有可绘制的数据列")
        return
    
    num_cols = len(columns_to_plot)
    num_rows = (num_cols + 2) // 3  # 每行3个图
    num_cols_per_row = min(3, num_cols)
    
    # 创建图形
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(16, 4 * num_rows))
    
    # 如果只有一个子图，axes 不是数组
    if num_rows == 1 and num_cols_per_row == 1:
        axes = [[axes]]
    elif num_rows == 1 or num_cols_per_row == 1:
        axes = axes.reshape(num_rows, num_cols_per_row) if num_rows > 1 else axes.reshape(1, -1)
    
    axes = axes.flatten()  # 展平数组方便遍历
    
    # 绘制每一列的数据
    for idx, col in enumerate(columns_to_plot):
        ax = axes[idx]
        data = df[col].values
        mean_val = data.mean()
        
        ax.plot(data, linewidth=1.5, alpha=0.8, color='steelblue', label='Data')
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.set_xlabel('Step')
        ax.set_ylabel(col)
        ax.set_title(f'{col}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 打印统计信息
        print(f"\n{col}:")
        print(f"  总数据点: {len(data)}")
        print(f"  平均值: {mean_val:.4f}")
        print(f"  最大值: {data.max():.4f}")
        print(f"  最小值: {data.min():.4f}")
        print(f"  标准差: {data.std():.4f}")
    
    # 隐藏多余的子图
    for idx in range(len(columns_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(log_file)
    if not output_dir:
        output_dir = '.'
    
    # 从日志文件名提取时间戳
    log_name = os.path.basename(log_file).replace('.txt', '')
    output_file = os.path.join(output_dir, f'{log_name}_all_metrics.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图像已保存到: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # 默认读取最新的日志文件
        log_dir = "log"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.startswith('training_log_') and f.endswith('.txt')]
            if log_files:
                log_files.sort(reverse=True)
                log_file = os.path.join(log_dir, log_files[0])
                print(f"读取最新日志文件: {log_file}")
            else:
                print(f"在 {log_dir} 目录中未找到日志文件")
                sys.exit(1)
        else:
            print(f"日志目录 {log_dir} 不存在")
            sys.exit(1)
    
    if os.path.exists(log_file):
        plot_reward_from_log(log_file)
    else:
        print(f"文件不存在: {log_file}")
        sys.exit(1)
