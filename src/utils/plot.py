import matplotlib.pyplot as plt
def plot_trajectories(trajs, save_path="batch_trajectories.png"):
    trajs_cpu = trajs.detach().cpu()  # 转回 CPU
    steps, batch, _ = trajs_cpu.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for b in range(batch):
        ax.plot(
            trajs_cpu[:, b, 0].numpy(),
            trajs_cpu[:, b, 1].numpy(),
            trajs_cpu[:, b, 2].numpy(),
            label=f'Traj {b}'
        )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Batch 3D Trajectories')
    ax.legend()
    plt.show()
    fig.savefig(save_path)
def plot_trajectories_grid(trajs, ncols=5, save_path="batch_trajectories_grid.png"):
    trajs_cpu = trajs.detach().cpu()  # 转回 CPU
    steps, batch, _ = trajs_cpu.shape
    
    nrows = (batch + ncols - 1) // ncols  # 计算行数
    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    
    for b in range(batch):
        ax = fig.add_subplot(nrows, ncols, b+1, projection='3d')
        ax.plot(
            trajs_cpu[:, b, 0].numpy(),
            trajs_cpu[:, b, 1].numpy(),
            trajs_cpu[:, b, 2].numpy(),
            label=f'Traj {b}'
        )
        ax.set_title(f'Traj {b}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path)