

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CoolingSystemSimulator:
    """
    斯特林制冷机老化模型（制冷时间双指数趋势强化版）
    核心指标：
    1. 稳态温度 (T_stable)：指数退化
    2. 制冷时间 (t_cool)：双指数退化（明显非线性）
    3. 温度波动 (sigma_T)：线性退化
    """
    def __init__(self, 
                 # 稳态温度参数（不变）
                 T_C0: float = 77.0,
                 T_CE: float = 5.72,
                 tau: float = 2781 / 5,
                 # ==================== 核心修改：制冷时间双指数参数（强化非线性） ====================
                 a: float = 0.05,        # 第一指数幅值（调小）
                 b: float = 8e-5,        # 第一指数系数（大幅放大，原≈8e-8）
                 c: float = 0.15,        # 第二指数幅值（调大）
                 d: float = 3e-4,        # 第二指数系数（大幅放大，原≈2.16e-6）
                 # 温度波动参数（不变）
                 sigma_T0: float = 0.3,
                 beta_T: float = 0.001 / 3600):
        # 稳态温度
        self.T_C0 = T_C0
        self.T_CE = T_CE
        self.tau = tau
        
        # 制冷时间（双指数强化）
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # 温度波动
        self.sigma_T0 = sigma_T0
        self.beta_T = beta_T

    def get_stable_temperature(self, t_hours: float) -> float:
        """稳态温度（指数退化）"""
        return self.T_C0 + self.T_CE * (1 - np.exp(-t_hours / self.tau))

    def get_cooling_time(self, t_hours: float) -> float:
        """制冷时间（双指数退化，强化非线性）"""
        return self.a * np.exp(self.b * t_hours) + self.c * np.exp(self.d * t_hours)

    def get_temperature_fluctuation(self, t_hours: float) -> float:
        """温度波动（线性增长）"""
        return self.sigma_T0 * (1 + self.beta_T * 3600 * t_hours)

# ==================== 按工作周期生成数据（1周期=8h，不变） ====================
def generate_single_simulation(total_cycles: int, cooler_params: Dict = None) -> pd.DataFrame:
    HOURS_PER_CYCLE = 8
    cooler_params = cooler_params or {}
    cooler = CoolingSystemSimulator(**cooler_params)
    
    work_cycles = np.arange(0, total_cycles + 1)
    time_hours = work_cycles * HOURS_PER_CYCLE
    
    data = {
        'work_cycle': work_cycles,
        'T_stable_K': [],
        't_cool_hours': [],
        'sigma_T_K': []
    }
    
    for t in time_hours:
        data['T_stable_K'].append(cooler.get_stable_temperature(t))
        data['t_cool_hours'].append(cooler.get_cooling_time(t))
        data['sigma_T_K'].append(cooler.get_temperature_fluctuation(t))
    
    return pd.DataFrame(data)

# ==================== 随机参数区间同步优化（保证随机组也有双指数趋势） ====================
def generate_random_params(num_groups: int, param_ranges: Dict = None) -> List[Dict]:
    # 随机参数区间同步强化双指数趋势
    default_ranges = {
        'T_C0': (75.0, 79.0),
        'T_CE': (4.0, 7.0),
        'tau': (450.0, 650.0),
        'a': (0.03, 0.08),      # 匹配优化后幅值
        'b': (6e-5, 1e-4),     # 匹配优化后指数系数
        'c': (0.10, 0.20),     # 匹配优化后幅值
        'd': (2e-4, 4e-4),     # 匹配优化后指数系数
        'sigma_T0': (0.2, 0.4),
        'beta_T': (2.5e-7, 3e-7)
    }
    param_ranges = param_ranges or default_ranges
    return [{p: random.uniform(min_v, max_v) for p, (min_v, max_v) in param_ranges.items()} for _ in range(num_groups)]

def generate_multi_simulation(total_cycles: int, num_groups: int, param_ranges: Dict = None) -> Dict[int, pd.DataFrame]:
    random_params_list = generate_random_params(num_groups, param_ranges)
    multi_data = {}
    for group_id, params in enumerate(random_params_list):
        df = generate_single_simulation(total_cycles, params)
        df['group_id'] = group_id
        multi_data[group_id] = df
    return multi_data

# ==================== 可视化（不变，X轴工作周期） ====================
def plot_simulation_data(df: pd.DataFrame, title: str = "制冷机退化仿真曲线", save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x = df['work_cycle']
    
    # 稳态温度（指数）
    axes[0].plot(x, df['T_stable_K'], 'b-', linewidth=2, label='稳态温度')
    axes[0].set_ylabel('稳态温度 (K)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('稳态温度（指数退化）')
    
    # 制冷时间（双指数，明显非线性）
    axes[1].plot(x, df['t_cool_hours'], 'r-', linewidth=2, label='制冷时间')
    axes[1].set_ylabel('制冷时间 (hours)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('制冷时间（双指数退化，已强化非线性）')
    
    # 温度波动（线性）
    axes[2].plot(x, df['sigma_T_K'], 'g-', linewidth=2, label='温度波动')
    axes[2].set_xlabel('工作周期数（1周期=8h）')
    axes[2].set_ylabel('温度波动标准差 (K)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_title('温度波动（线性退化）')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 数据保存（不变） ====================
def save_simulation_data(data: Dict[int, pd.DataFrame], save_dir: str = "./cooler_simulation_results"):
    os.makedirs(save_dir, exist_ok=True)
    pd.concat(data.values(), ignore_index=True).to_csv(os.path.join(save_dir, "all_simulation.csv"), index=False)
    for gid, df in data.items():
        df.to_csv(os.path.join(save_dir, f"simulation_group_{gid}.csv"), index=False)

# ==================== 前30/后30数据提取（不变） ====================
def extract_head_tail_data(df: pd.DataFrame, n_points: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total = len(df)
    return df.head(min(n_points, total)).copy(), df.tail(min(n_points, total)).copy()

def save_head_tail_data(data: Dict[int, pd.DataFrame], save_dir: str = "./cooler_simulation_results", n_points: int = 30):
    all_head, all_tail = [], []
    for gid, df in data.items():
        head, tail = extract_head_tail_data(df, n_points)
        head.to_csv(os.path.join(save_dir, f"group_{gid}_head_{n_points}.csv"), index=False)
        tail.to_csv(os.path.join(save_dir, f"group_{gid}_tail_{n_points}.csv"), index=False)
        all_head.append(head)
        all_tail.append(tail)
    pd.concat(all_head).to_csv(os.path.join(save_dir, f"all_head_{n_points}.csv"), index=False)
    pd.concat(all_tail).to_csv(os.path.join(save_dir, f"all_tail_{n_points}.csv"), index=False)
    print(f"✅ 前{n_points}、后{n_points}个工作周期数据已保存！")

# ==================== 主程序 ====================
if __name__ == "__main__":
    TOTAL_CYCLES = 800   # 总工作周期（8h/周期，共6400h）
    NUM_GROUPS = 10       # 随机参数组数
    
    # 1. 单组仿真
    print("=== 单组参数仿真（双指数趋势已强化） ===")
    custom_params = {
        'T_C0':77, 'T_CE':5.72, 'tau':556.2,
        'a':0.05, 'b':8e-5, 'c':0.15, 'd':3e-4,  # 强化双指数核心参数
        'sigma_T0':0.3, 'beta_T':2.77e-7
    }
    single_df = generate_single_simulation(TOTAL_CYCLES, custom_params)
    print("单组数据前5行：")
    print(single_df.head())
    plot_simulation_data(single_df, "单组参数-制冷机退化曲线（双指数强化版）")
    
    # 保存单组关键数据
    single_head, single_tail = extract_head_tail_data(single_df)
    os.makedirs("./cooler_simulation_results", exist_ok=True)
    single_head.to_csv("./cooler_simulation_results/single_head_30.csv", index=False)
    single_tail.to_csv("./cooler_simulation_results/single_tail_30.csv", index=False)

    # 2. 多组随机参数仿真
    print("\n=== 多组随机参数仿真 ===")
    multi_data = generate_multi_simulation(TOTAL_CYCLES, NUM_GROUPS)
    plot_simulation_data(multi_data[0], "随机参数组0-制冷机退化曲线（双指数强化版）")
    
    # 保存数据
    save_simulation_data(multi_data)
    save_head_tail_data(multi_data)
    
    print("\n🎉 制冷时间双指数趋势强化完成！")