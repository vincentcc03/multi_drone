import matplotlib.pyplot as plt
import os

log_path = 'log/training_log_20251021_202202.txt'
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'reward_curve.png')

rewards = []
with open(log_path, 'r') as f:
    for line in f:
        if line.startswith('max_progress') or line.startswith('Training Log') or line.startswith('-') or line.strip() == '':
            continue
        parts = line.strip().split(',')
        if len(parts) > 1:
            try:
                reward = float(parts[1])
                rewards.append(reward)
            except ValueError:
                continue

plt.figure(figsize=(10, 5))
plt.plot(rewards, label='env1_Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path)
print(f"Reward curve saved to {save_path}")