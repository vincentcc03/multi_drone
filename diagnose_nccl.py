import os
import subprocess
import sys

print("="*60)
print("NCCL冲突诊断")
print("="*60)

# 1. 检查LD_PRELOAD是否生效
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', '未设置')}")

# 2. 查看所有可能的nccl库
print("\n所有libnccl.so.2文件:")
for root, dirs, files in os.walk('/'):
    for file in files:
        if file.startswith('libnccl') and file.endswith('.so.2'):
            full_path = os.path.join(root, file)
            print(f"  {full_path}")
            # 检查文件信息
            try:
                result = subprocess.run(['file', full_path], capture_output=True, text=True)
                print(f"    {result.stdout.strip()}")
                
                # 检查NCCL版本
                result = subprocess.run(['strings', full_path, '|', 'grep', 'NCCL_VERSION', '|', 'head', '-1'], 
                                      shell=True, capture_output=True, text=True)
                if result.stdout:
                    print(f"    版本: {result.stdout.strip()}")
            except:
                pass
    # 限制深度，避免无限循环
    if root.count('/') > 3:
        dirs[:] = []

print("\n" + "="*60)
