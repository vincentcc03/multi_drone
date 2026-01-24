#!/usr/bin/env python3
"""
分析log文件夹中所有txt文件的运行时长和行数，并绘制关系图表
"""

import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filepath):
    """
    解析log文件，返回运行时长（秒）和行数
    使用文件修改时间和creation time来计算运行时长
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # 获取文件的修改时间和创建时间
        stat_info = os.stat(filepath)
        mtime = stat_info.st_mtime  # 修改时间
        
        # 从文件名提取创建时间
        basename = os.path.basename(filepath)
        # 格式: training_log_20260115_224903.txt
        date_match = re.search(r'training_log_(\d{8})_(\d{6})', basename)
        if not date_match:
            return None
        
        date_str = date_match.group(1)  # 20260115
        start_time_hms = date_match.group(2)  # 224903
        
        # 转换成标准格式
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = start_time_hms[:2]
        minute = start_time_hms[2:4]
        second = start_time_hms[4:6]
        
        start_datetime = datetime.strptime(f"{year}-{month}-{day} {hour}:{minute}:{second}", 
                                           '%Y-%m-%d %H:%M:%S')
        end_datetime = datetime.fromtimestamp(mtime)
        
        # 计算运行时长（秒）
        duration = (end_datetime - start_datetime).total_seconds()
        
        if duration <= 0:
            return None
        
        # 计算数据行（跳过header行）
        header_lines = 0
        for i, line in enumerate(lines):
            if line.startswith('max_progress') or line.startswith('Progress'):
                header_lines = i + 1
                break
        
        # 如果没有找到header，使用前15行作为header
        if header_lines == 0:
            header_lines = 15
        
        data_lines = max(len(lines) - header_lines, 1)
        
        return duration, data_lines
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    log_dir = './log'
    
    # 收集数据
    data = []
    filenames = []
    
    # 列出所有txt文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
    
    print(f"Found {len(log_files)} log files")
    print("Processing files...")
    
    for filename in sorted(log_files):
        filepath = os.path.join(log_dir, filename)
        result = parse_log_file(filepath)
        
        if result:
            duration, lines = result
            data.append((duration, lines))
            filenames.append(filename)
            print(f"✓ {filename}: {duration:.1f}s, {lines} lines")
        else:
            print(f"✗ {filename}: Failed to parse")
    
    if not data:
        print("No valid data to plot!")
        return
    
    # 解包数据
    durations = np.array([d[0] for d in data])
    line_counts = np.array([d[1] for d in data])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 散点图
    scatter = ax.scatter(line_counts, durations, s=100, alpha=0.6, c=range(len(data)), cmap='viridis')
    
    # 添加趋势线
    z = np.polyfit(line_counts, durations, 1)
    p = np.poly1d(z)
    ax.plot(line_counts, p(line_counts), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.6f}x+{z[1]:.1f}')
    
    # 标签和标题
    ax.set_xlabel('Number of Lines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Log File Analysis: Duration vs Line Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('File Index', fontsize=10)
    
    # 保存图表
    output_file = 'log_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_file}")
    
    # 显示统计信息
    print(f"\nStatistics:")
    print(f"  Total files analyzed: {len(data)}")
    print(f"  Duration range: {durations.min():.1f}s - {durations.max():.1f}s")
    print(f"  Average duration: {durations.mean():.1f}s")
    print(f"  Line count range: {line_counts.min()} - {line_counts.max()}")
    print(f"  Average line count: {line_counts.mean():.0f}")
    print(f"  Correlation: {np.corrcoef(line_counts, durations)[0, 1]:.4f}")
    
    plt.show()

if __name__ == '__main__':
    main()
