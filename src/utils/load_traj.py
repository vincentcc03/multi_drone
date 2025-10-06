import numpy as np
from src.utils.read_yaml import load_config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_coefficients_from_files():
    """
    直接从文件加载多项式系数并拼接
    """
    Coeffx = np.zeros((2, 8))
    Coeffy = np.zeros((2, 8))
    Coeffz = np.zeros((2, 8)) 
    
    for k in range(2):
        Coeffx[k, :] = np.load(f'document/Reference_traj_4/coeffx{str(k+1)}.npy')
        Coeffy[k, :] = np.load(f'document/Reference_traj_4/coeffy{str(k+1)}.npy')
        Coeffz[k, :] = np.load(f'document/Reference_traj_4/coeffz{str(k+1)}.npy')

    return Coeffx, Coeffy, Coeffz

def polytraj(coeff, time, t_switch):
    """多项式轨迹计算函数"""
    t = time - t_switch
    # 8阶多项式轨迹计算
    pos = coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5 + coeff[6]*t**6 + coeff[7]*t**7
    vel = coeff[1] + 2*coeff[2]*t + 3*coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4 + 6*coeff[6]*t**5 + 7*coeff[7]*t**6
    acc = 2*coeff[2] + 6*coeff[3]*t + 12*coeff[4]*t**2 + 20*coeff[5]*t**3 + 30*coeff[6]*t**4 + 42*coeff[7]*t**5
    jerk = 6*coeff[3] + 24*coeff[4]*t + 60*coeff[5]*t**2 + 120*coeff[6]*t**3 + 210*coeff[7]*t**4
    snap = 24*coeff[4] + 120*coeff[5]*t + 360*coeff[6]*t**2 + 840*coeff[7]*t**3
    
    return pos, vel, acc, jerk, snap

def minisnap_load_circle(time, config_path="traj_config.yaml"):
    """
    独立的圆形轨迹生成+函数
    现在不需要传入Coeffx等参数，直接从文件读取
    """
    # 加载系数和配置
    Coeffx, Coeffy, Coeffz = load_coefficients_from_files()
    config = load_config(config_path)
    
    # 从配置中获取参数
    ml = config["ml"]          # 负载质量 (kg)
    g = config["g"]            # 重力加速度 (m/s²)
    nxl = config["nxl"]        # 负载状态向量维度
    nul = config["nul"]        # 控制输入向量维度
    nWl = config["nWl"]        # 力和力矩向量维度
    nq = config["nq"]          # 四元数相关参数
    ez = np.array(config["ez"]).reshape(3, 1)  # Z轴单位向量
    S_rg = np.array(config["S_rg"])  # 空间变换矩阵
    
    t_switch = 0
    t1 = 2.4
    
    # 选择轨迹段
    if time < t1:
        ref_px, ref_vx, ref_ax, _, _ = polytraj(Coeffx[0,:], time, t_switch)
        ref_py, ref_vy, ref_ay, _, _ = polytraj(Coeffy[0,:], time, t_switch)
        ref_pz, ref_vz, ref_az, _, _ = polytraj(Coeffz[0,:], time, t_switch)
    else:
        ref_px, ref_vx, ref_ax, _, _ = polytraj(Coeffx[1,:], time, t1 + t_switch)
        ref_py, ref_vy, ref_ay, _, _ = polytraj(Coeffy[1,:], time, t1 + t_switch)
        ref_pz, ref_vz, ref_az, _, _ = polytraj(Coeffz[1,:], time, t1 + t_switch)
    
    # 构建状态和控制向量
    ref_p = np.vstack((ref_px, ref_py, ref_pz))
    ref_v = np.vstack((ref_vx, ref_vy, ref_vz))
    ref_q = np.array([[1, 0, 0, 0]]).T
    ref_w = np.zeros((3, 1))
    ref_xl = np.vstack((ref_p, ref_v, ref_q, ref_w)).flatten()
    
    ref_a = np.vstack((ref_ax, ref_ay, ref_az))
    ref_Fl = ml * (ref_a + g * ez)
    ref_ml = S_rg @ (ml * g * ez)
    ref_nv = np.zeros((3 * nq - nWl, 1))
    ref_ul = np.vstack((ref_Fl, ref_ml, ref_nv)).flatten()
    ref_Wl = np.vstack((ref_Fl, ref_ml)).flatten()
    
    # 向心加速度计算
    delta = 1e-8
    ac = (ref_vx * ref_ay - ref_vy * ref_ax) / (ref_vx**2 + ref_vy**2 + delta) * np.array([[-ref_vy], [ref_vx]])
    
    return ref_xl, ref_ul, ref_p, ref_Wl, ac

def generate_complete_trajectory(config_path="traj_config.yaml"):
    """
    生成完整的参考轨迹，所有轨迹变量均为二维数组
    """
    config = load_config(config_path)
    horizon = config["horizon"]
    dt = config["dt"]
    nxl = config["nxl"]
    nul = config["nul"]
    nWl = config["nWl"]

    # 初始化为二维数组
    Ref_xl = np.zeros((horizon + 1, nxl))
    Ref_ul = np.zeros((horizon, nul))
    Ref_pl = np.zeros((horizon + 1, 3))
    Ref_Wl = np.zeros((horizon, nWl))
    Ref_ac = np.zeros((horizon + 1, 2))
    Time = np.zeros(horizon + 1)

    time = 0
    for k in range(horizon):
        Time[k] = time
        ref_xl, ref_ul, ref_p, ref_Wl, ac = minisnap_load_circle(time, config_path)

        Ref_xl[k, :] = ref_xl
        Ref_ul[k, :] = ref_ul
        Ref_Wl[k, :] = ref_Wl
        Ref_pl[k, :] = ref_p.flatten()
        Ref_ac[k, :] = ac.flatten()

        time += dt

    # 最终时间点
    Time[horizon] = time
    ref_xl, ref_ul, ref_p, ref_Wl, ac = minisnap_load_circle(time, config_path)
    Ref_xl[horizon, :] = ref_xl
    Ref_pl[horizon, :] = ref_p.flatten()
    Ref_ac[horizon, :] = ac.flatten()

    return {
        'Ref_xl': Ref_xl,      # (horizon+1, nxl)
        'Ref_ul': Ref_ul,      # (horizon, nul)
        'Ref_pl': Ref_pl,      # (horizon+1, 3)
        'Ref_Wl': Ref_Wl,      # (horizon, nWl)
        'Ref_ac': Ref_ac,      # (horizon+1, 2)
        'Time': Time           # (horizon+1,)
    }

def plot_3d_trajectory(trajectory, config_path="traj_config.yaml"):
    """
    绘制三维轨迹图

    参数:
    trajectory: generate_complete_trajectory 返回的轨迹字典
    config_path: 配置文件路径
    """
    config = load_config(config_path)

    # 获取位置数据
    x = trajectory['Ref_pl'][:, 0]
    y = trajectory['Ref_pl'][:, 1]
    z = trajectory['Ref_pl'][:, 2]
    time = trajectory['Time']

    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=2, label='轨迹')
    ax1.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='起点')
    ax1.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='x', label='终点')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('负载三维轨迹')
    ax1.legend()
    ax1.grid(True)

    # 各个平面的投影
    ax2 = fig.add_subplot(322)
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.scatter(x[0], y[0], c='g', s=50)
    ax2.scatter(x[-1], y[-1], c='r', s=50)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面投影')
    ax2.grid(True)
    ax2.axis('equal')

    ax3 = fig.add_subplot(324)
    ax3.plot(x, z, 'b-', linewidth=2)
    ax3.scatter(x[0], z[0], c='g', s=50)
    ax3.scatter(x[-1], z[-1], c='r', s=50)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ平面投影')
    ax3.grid(True)

    ax4 = fig.add_subplot(326)
    ax4.plot(y, z, 'b-', linewidth=2)
    ax4.scatter(y[0], z[0], c='g', s=50)
    ax4.scatter(y[-1], z[-1], c='r', s=50)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('YZ平面投影')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    # 单独的时间序列图
    fig2, (ax5, ax6, ax7) = plt.subplots(3, 1, figsize=(12, 8))

    ax5.plot(time, x, 'r-', label='X')
    ax5.plot(time, y, 'g-', label='Y')
    ax5.set_ylabel('位置 (m)')
    ax5.set_title('X和Y位置随时间变化')
    ax5.legend()
    ax5.grid(True)

    ax6.plot(time, z, 'b-')
    ax6.set_ylabel('Z位置 (m)')
    ax6.set_title('Z位置随时间变化')
    ax6.grid(True)

    # 速度大小
    vx = np.gradient(x, time)
    vy = np.gradient(y, time)
    vz = np.gradient(z, time)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    ax7.plot(time, speed, 'm-')
    ax7.set_xlabel('时间 (s)')
    ax7.set_ylabel('速度 (m/s)')
    ax7.set_title('速度大小随时间变化')
    ax7.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印轨迹信息
    print(f"轨迹总时间: {time[-1]:.2f} 秒")
    print(f"轨迹点数: {len(time)}")
    print(f"X范围: [{x.min():.2f}, {x.max():.2f}] m")
    print(f"Y范围: [{y.min():.2f}, {y.max():.2f}] m")
    print(f"Z范围: [{z.min():.2f}, {z.max():.2f}] m")
    print(f"最大速度: {speed.max():.2f} m/s")

# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("开始生成完整轨迹...")
    print("=" * 60)
    
    # 生成完整轨迹
    trajectory = generate_complete_trajectory()
    
    print(f"✓ 轨迹生成完成！")
    print(f"完整状态轨迹维度: {trajectory['Ref_xl'].shape}")
    print(f"控制输入轨迹维度: {trajectory['Ref_ul'].shape}")
    print(f"位置轨迹维度: {trajectory['Ref_pl'].shape}")
    print(f"力和力矩轨迹维度: {trajectory['Ref_Wl'].shape}")
    print(f"向心加速度轨迹维度: {trajectory['Ref_ac'].shape}")
    print(f"时间序列长度: {trajectory['Time'].shape}")
    
    print("\n" + "=" * 60)
    print("轨迹数据详细信息:")
    print("=" * 60)
    
    # 打印时间信息
    print(f"总仿真时间: {trajectory['Time'][-1]:.3f} 秒")
    print(f"时间步长: {trajectory['Time'][1] - trajectory['Time'][0]:.4f} 秒")
    print(f"轨迹点数量: {len(trajectory['Time'])}")
    
    # 打印位置轨迹信息
    print(f"\n📍 位置轨迹范围:")
    print(f"  X: [{trajectory['Ref_pl'][:, 0].min():.3f}, {trajectory['Ref_pl'][:, 0].max():.3f}] m")
    print(f"  Y: [{trajectory['Ref_pl'][:, 1].min():.3f}, {trajectory['Ref_pl'][:, 1].max():.3f}] m")
    print(f"  Z: [{trajectory['Ref_pl'][:, 2].min():.3f}, {trajectory['Ref_pl'][:, 2].max():.3f}] m")
    
    # 打印初始状态
    print(f"\n🚀 初始状态 (t=0):")
    print(f"  位置: [{trajectory['Ref_xl'][0, 0]:.3f}, {trajectory['Ref_xl'][0, 1]:.3f}, {trajectory['Ref_xl'][0, 2]:.3f}] m")
    print(f"  速度: [{trajectory['Ref_xl'][0, 3]:.3f}, {trajectory['Ref_xl'][0, 4]:.3f}, {trajectory['Ref_xl'][0, 5]:.3f}] m/s")
    print(f"  四元数: [{trajectory['Ref_xl'][0, 6]:.3f}, {trajectory['Ref_xl'][0, 7]:.3f}, {trajectory['Ref_xl'][0, 8]:.3f}, {trajectory['Ref_xl'][0, 9]:.3f}]")
    print(f"  角速度: [{trajectory['Ref_xl'][0, 10]:.3f}, {trajectory['Ref_xl'][0, 11]:.3f}, {trajectory['Ref_xl'][0, 12]:.3f}] rad/s")
    
    # 打印终止状态
    print(f"\n🏁 终止状态 (t={trajectory['Time'][-1]:.3f}s):")
    print(f"  位置: [{trajectory['Ref_xl'][-1, 0]:.3f}, {trajectory['Ref_xl'][-1, 1]:.3f}, {trajectory['Ref_xl'][-1, 2]:.3f}] m")
    print(f"  速度: [{trajectory['Ref_xl'][-1, 3]:.3f}, {trajectory['Ref_xl'][-1, 4]:.3f}, {trajectory['Ref_xl'][-1, 5]:.3f}] m/s")
    print(f"  四元数: [{trajectory['Ref_xl'][-1, 6]:.3f}, {trajectory['Ref_xl'][-1, 7]:.3f}, {trajectory['Ref_xl'][-1, 8]:.3f}, {trajectory['Ref_xl'][-1, 9]:.3f}]")
    print(f"  角速度: [{trajectory['Ref_xl'][-1, 10]:.3f}, {trajectory['Ref_xl'][-1, 11]:.3f}, {trajectory['Ref_xl'][-1, 12]:.3f}] rad/s")
    
    # 打印一些中间时刻的轨迹点
    print(f"\n📊 轨迹采样点 (每10%显示一个点):")
    step_size = len(trajectory['Time']) // 100
    for i in range(0, len(trajectory['Time']), step_size):
        if i < len(trajectory['Time']):
            t = trajectory['Time'][i]
            pos = trajectory['Ref_pl'][i]
            print(f"  t={t:6.3f}s: 位置=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] m")
    
    # 计算速度信息
    velocities = trajectory['Ref_xl'][:, 3:6]  # 提取速度分量
    speeds = np.linalg.norm(velocities, axis=1)  # 计算速度大小
    
    print(f"\n🏃 速度统计:")
    print(f"  最大速度: {speeds.max():.3f} m/s")
    print(f"  平均速度: {speeds.mean():.3f} m/s")
    print(f"  最小速度: {speeds.min():.3f} m/s")
    
    # 控制输入信息
    if trajectory['Ref_ul'].size > 0:
        print(f"\n🎮 控制输入统计:")
        print(f"  力输入范围: [{trajectory['Ref_ul'][:, 0:3].min():.3f}, {trajectory['Ref_ul'][:, 0:3].max():.3f}] N")
        print(f"  力矩输入范围: [{trajectory['Ref_ul'][:, 3:6].min():.3f}, {trajectory['Ref_ul'][:, 3:6].max():.3f}] N⋅m")
    
    print(f"\n" + "=" * 60)
    print("是否绘制三维轨迹图？")
    print("=" * 60)
    
    # 询问是否绘制图形
    try:
        response = input("输入 'y' 或 'yes' 来绘制轨迹图 (默认: n): ").lower().strip()
        if response in ['y', 'yes']:
            print("正在绘制三维轨迹图...")
            plot_3d_trajectory(trajectory)
        else:
            print("跳过轨迹绘制")
    except KeyboardInterrupt:
        print("\n跳过轨迹绘制")
    
    print(f"\n✅ 轨迹生成和分析完成！")