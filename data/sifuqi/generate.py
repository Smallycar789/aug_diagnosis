"""
伺服控制精度退化仿真 v8 — 科学验证版
=====================================
基于 DC 电机 + PID 控制器 + 退化物理模型的科学仿真。

理论公式:
  电机电气:  V = R·i + L·di/dt + Kb·ω
  电机机械:  J·dω/dt = Kt·i - B·ω - Tf·sign(ω)
  PID 控制:  u = Kp·e + Ki·∫e dt + Kd·de/dt

退化规律:
  R(t) = R₀·(1 + α_R·t)           电阻热老化 (线性增长)
  Kt(t) = Kt₀·exp(-β_K·t)          磁钢退磁 (指数衰减)
  B(t) = B₀·(1 + γ_B·t^1.5)        轴承磨损 (加速增长)
  Tf(t) = Tf₀ + ΔTf·(1-exp(-λ·t))  摩擦增大 (指数趋近)

稳态误差: e_ss ≈ deadzone + Tf/Kp
跟踪误差: e_track ≈ A·ω·τ  (τ = J/B_eff, 一阶近似)

噪声模型:
  传感器量化:  σ_q = resolution / √12
  热噪声:      σ_th = sqrt(4kB·T·R·BW)
  振动:        含工频50Hz + 谐波 + 宽带随机
  MTF退化:     MTF = MTF₀·exp(-(σ_jitter/σ_crit)²)
"""
import numpy as np
import csv, json, os

OUT_DIR = r"D:\2024BUAAyanyi\实验室研一\航天测控-云台\数字化云台\simulation_data\sifuqi"
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = [
    "timestamp", "azimuth_cmd", "azimuth_actual", "azimuth_error",
    "pitch_cmd", "pitch_actual", "pitch_error",
    "azimuth_current", "pitch_current", "temperature", "vibration", "mtf", "label"
]

# ===================== 科学参数 =====================
# DC 电机标称参数 (二自由度云台典型值)
J_AZ, J_EL = 0.004, 0.003      # 转动惯量 (kg·m²)
L_AZ, L_EL = 0.002, 0.0015     # 电感 (H)
R0_AZ, R0_EL = 2.5, 2.0       # 标称电阻 (Ω)
Kt0_AZ, Kt0_EL = 0.45, 0.38   # 标称力矩常数 (N·m/A)
Kb0_AZ, Kb0_EL = 0.45, 0.38   # 标称反电势常数 (V·s/rad) = Kt (SI单位)
B0_AZ, B0_EL = 0.0008, 0.0006 # 标称阻尼系数 (N·m·s/rad)
Tf0_AZ, Tf0_EL = 0.003, 0.002 # 标称库仑摩擦 (N·m)

# PID — 中等增益, 让机械退化在控制中显现
PID_NORMAL = {
    "az": {"Kp_pos": 25, "Kp_vel": 5.0, "Ki_vel": 20},
    "el": {"Kp_pos": 20, "Kp_vel": 4.0, "Ki_vel": 16},
}

# 退化系数 — 放大差异, 让各级别明显区分
DEG_COEF = {
    "normal":   {"alpha_R": 0.00, "beta_K": 0.000, "gamma_B": 0.000, "delta_Tf": 0.000, "tau_Tf": 1e9, "hours": 0},
    "mild":     {"alpha_R": 0.15, "beta_K": 0.030, "gamma_B": 0.006, "delta_Tf": 0.015, "tau_Tf": 600,  "hours": 600},
    "moderate": {"alpha_R": 0.40, "beta_K": 0.080, "gamma_B": 0.018, "delta_Tf": 0.040, "tau_Tf": 400,  "hours": 1800},
    "severe":   {"alpha_R": 0.90, "beta_K": 0.180, "gamma_B": 0.045, "delta_Tf": 0.100, "tau_Tf": 250,  "hours": 3500},
}

# 噪声模型参数
SENSOR_RESOLUTION = 0.0003       # 编码器分辨率 (deg) ~ 18-bit
THERMAL_R = 1000                  # 等效热噪声电阻 (Ω)
BANDWIDTH = 100                   # 系统带宽 (Hz)
kB = 1.38e-23                     # 玻尔兹曼常数
T_AMB = 298                       # 环境温度 (K)


def degrade_params(degradation, axis="az"):
    """根据退化等级计算当前电机参数"""
    J = J_AZ if axis == "az" else J_EL
    L = L_AZ if axis == "az" else L_EL
    R0 = R0_AZ if axis == "az" else R0_EL
    Kt0 = Kt0_AZ if axis == "az" else Kt0_EL
    Kb0 = Kb0_AZ if axis == "az" else Kb0_EL
    B0 = B0_AZ if axis == "az" else B0_EL
    Tf0 = Tf0_AZ if axis == "az" else Tf0_EL

    c = DEG_COEF[degradation]
    t = c["hours"] / 1000.0  # 归一化时间 (1000h单位)

    R = R0 * (1 + c["alpha_R"] * t)
    Kt = Kt0 * np.exp(-c["beta_K"] * t)
    Kb = Kb0 * np.exp(-c["beta_K"] * t)  # Kt ≈ Kb (SI)
    B = B0 * (1 + c["gamma_B"] * t**1.5)
    Tf = Tf0 + c["delta_Tf"] * (1 - np.exp(-t * 1000 / c["tau_Tf"]))

    return {
        "J": J, "L": L, "R": R, "Kt": Kt, "Kb": Kb, "B": B, "Tf": Tf,
        "R0": R0, "Kt0": Kt0, "B0": B0, "Tf0": Tf0,
    }


def generate_servo(duration=60.0, dt=0.001, degradation="normal", seed=42):
    """
    基于 DC 电机 + PID + 噪声模型的高精度仿真
    dt=1ms for accurate PID simulation
    """
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    t = np.arange(n) * dt

    # ---- 指令轨迹 (大幅摆动, 清楚展示退化) ----
    az_cmd = 45.0 * np.sin(2 * np.pi * t / 12.0) + 8.0 * np.sin(2 * np.pi * t / 3.0)
    el_cmd = 28.0 * np.sin(2 * np.pi * t / 9.0 + 0.8) + 5.0 * np.cos(2 * np.pi * t / 3.5)

    # 阶跃段
    step_int = int(14.0 / dt)
    step_dur = int(2.0 / dt)
    is_step = np.zeros(n, dtype=bool)
    for i in range(step_int, n, step_int):
        end = min(i + step_dur, n)
        az_cmd[i:end] = rng.uniform(-45, 45)
        el_cmd[i:end] = rng.uniform(-28, 28)
        is_step[i:end] = True

    # ---- 噪声模型 ----
    # 量化噪声: σ_q = resolution / √12
    sigma_quant = SENSOR_RESOLUTION / np.sqrt(12)  # deg
    # 热噪声: σ_th = sqrt(4*kB*T*R*BW) → 转换为角度 (通过 Kt)
    sigma_thermal = np.sqrt(4 * kB * T_AMB * THERMAL_R * BANDWIDTH) * 0.1  # 折算到角度 deg
    # 综合传感器噪声
    sigma_sensor = np.sqrt(sigma_quant**2 + sigma_thermal**2)

    def generate_noise(n_samples, axis_seed):
        """多源噪声叠加"""
        local = np.random.default_rng(axis_seed)
        # 白噪声 (量化 + 热)
        white = local.normal(0, sigma_sensor, n_samples)
        # 1/f 闪烁噪声
        flicker = np.cumsum(local.normal(0, sigma_sensor * 0.3, n_samples)) * 0.05
        # 工频干扰: 50Hz + 谐波
        mains = 0.0005 * np.sin(2 * np.pi * 50 * t) + 0.0002 * np.sin(2 * np.pi * 100 * t)
        # 脉冲噪声 (偶发)
        impulse = np.zeros(n_samples)
        impulse_pos = local.choice(n_samples, size=int(n_samples * 0.001), replace=False)
        impulse[impulse_pos] = local.normal(0, sigma_sensor * 5, len(impulse_pos))
        return white + flicker + mains + impulse

    noise_az = generate_noise(n, seed + 300)
    noise_el = generate_noise(n, seed + 400)

    # ---- PID + 电机仿真 (全角度制, deg) ----
    def axis(cmd, deg_params, pid_params, noise, axis_seed):
        local = rng if axis_seed == 0 else np.random.default_rng(seed + axis_seed)
        J, L, R, Kt, Kb, B, Tf = [deg_params[k] for k in
            ["J", "L", "R", "Kt", "Kb", "B", "Tf"]]

        Kp_p, Kp_v, Ki_v = [pid_params[k] for k in
            ["Kp_pos", "Kp_vel", "Ki_vel"]]

        pos = np.zeros(n)
        vel = np.zeros(n)    # deg/s
        curr = np.zeros(n)
        pos[0] = cmd[0]

        vel_int = 0.0

        for k in range(1, n):
            # 位置环 P (deg -> deg/s)
            pos_err = cmd[k] - pos[k - 1]
            vel_cmd = Kp_p * pos_err

            # 速度环 PI (deg/s -> A)
            vel_err = vel_cmd - vel[k - 1]
            vel_int += vel_err * dt
            vel_int = np.clip(vel_int, -100, 100)
            i_cmd = Kp_v * vel_err + Ki_v * vel_int

            # 电流饱和
            i_cmd = np.clip(i_cmd, -3.0, 3.0)
            i_actual = i_cmd + local.normal(0, 0.01)

            # 电机机械: 力矩→角加速度 (rad -> deg/s²)
            # T = Kt*i (N·m),  J*α = T - B*ω - Tf
            # α(rad/s²) = (Kt*i - B*ω_rad - Tf) / J
            omega_rad = vel[k - 1] * np.pi / 180.0  # deg/s → rad/s
            T_motor = Kt * i_actual
            T_damp = B * omega_rad
            T_fric = Tf * np.sign(omega_rad) if abs(omega_rad) > 1e-6 else 0
            alpha_rad = (T_motor - T_damp - T_fric) / J
            alpha_deg = alpha_rad * 180.0 / np.pi  # rad/s² → deg/s²

            vel[k] = vel[k - 1] + alpha_deg * dt
            vel[k] += local.normal(0, 0.005)
            pos[k] = pos[k - 1] + vel[k] * dt + noise[k]
            curr[k] = i_actual

        return pos, vel, curr

    deg_az = degrade_params(degradation, "az")
    deg_el = degrade_params(degradation, "el")

    # PID 参数随运行时间退化 (驱动器老化)
    pid_degrade = 1.0 / (1.0 + 0.0005 * DEG_COEF[degradation]["hours"])
    pid_az = {k: v * pid_degrade for k, v in PID_NORMAL["az"].items()}
    pid_el = {k: v * pid_degrade for k, v in PID_NORMAL["el"].items()}

    az_pos, az_vel, az_cur = axis(az_cmd, deg_az, pid_az, noise_az, 100)
    el_pos, el_vel, el_cur = axis(el_cmd, deg_el, pid_el, noise_el, 200)

    az_err = az_cmd - az_pos
    el_err = el_cmd - el_pos

    # ---- 降采样到 100Hz 输出 ----
    ds = 10  # 1ms -> 10ms
    n_out = n // ds
    idx = np.arange(0, n, ds)[:n_out]
    t_out = t[idx]
    az_c_out = az_cmd[idx]
    az_p_out = az_pos[idx]
    az_e_out = az_err[idx]
    el_c_out = el_cmd[idx]
    el_p_out = el_pos[idx]
    el_e_out = el_err[idx]
    is_step_out = is_step[idx]

    # ---- 稳态误差 ----
    settle = int(1.2 / dt) // ds
    tail = int(0.6 / dt) // ds
    step_int_ds = step_int // ds
    s_az, s_el = [], []
    for i in range(step_int_ds, n_out - settle, step_int_ds):
        s, e = i + settle, min(i + settle + tail, n_out)
        if e > s:
            s_az.append(np.mean(np.abs(az_e_out[s:e])))
            s_el.append(np.mean(np.abs(el_e_out[s:e])))
    s_az = float(np.mean(s_az)) if s_az else 0.0
    s_el = float(np.mean(s_el)) if s_el else 0.0

    # ---- 跟踪误差 (纯正弦段) ----
    settle_after = int(3.0 / dt) // ds
    scan = ~is_step_out.copy()
    for i in range(step_int_ds, n_out - settle_after, step_int_ds):
        scan[i:i + step_dur // ds + settle_after] = False
    t_az = float(np.max(np.abs(az_e_out[scan]))) if np.any(scan) else 0.0
    t_el = float(np.max(np.abs(el_e_out[scan]))) if np.any(scan) else 0.0

    # ---- 传感器数据 ----
    I_az = np.abs(az_cur[idx]) + rng.normal(0, 0.01, n_out)
    I_el = np.abs(el_cur[idx]) + rng.normal(0, 0.01, n_out)

    # 温度模型: T = T_amb + ΔT_R + ΔT_friction
    T_amb = 25
    T_resistive = np.cumsum(np.clip(I_az**2 * deg_az["R"] + I_el**2 * deg_el["R"], 0, 8) * 0.015) * (dt * ds)
    T_friction = np.cumsum(np.abs(az_vel[idx]) * deg_az["Tf"] * 5 + np.abs(el_vel[idx]) * deg_el["Tf"] * 5) * (dt * ds)
    temp = np.clip(T_amb + T_resistive + T_friction + rng.normal(0, 0.15, n_out), 20, 85)

    # 振动模型: 机械 + 电磁 + 轴承
    vib_mech = (np.abs(np.gradient(az_vel[idx])) + np.abs(np.gradient(el_vel[idx]))) * 0.002
    vib_bearing = 0.001 * np.sqrt(deg_az["B"] / deg_az["B0"]) * np.sin(2 * np.pi * t_out * (30 + rng.random() * 5))
    vib_em = 0.0005 * np.sin(2 * np.pi * 50 * t_out) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t_out))
    vib_sensor = rng.normal(0.001, 0.0003, n_out)
    if degradation in ("moderate", "severe"):
        vib_bearing *= 3
        vib_em *= 2
    vib = np.clip(vib_mech + vib_bearing + vib_em + vib_sensor, 0, 0.3)

    # MTF 模型: MTF = MTF₀ · exp(-(σ_jitter / σ_crit)²)
    sigma_jitter = np.sqrt(np.mean(az_e_out**2 + el_e_out**2))
    sigma_crit = 0.5  # 临界抖动
    MTF0 = {"normal": 0.95, "mild": 0.85, "moderate": 0.70, "severe": 0.50}[degradation]
    mtf_base = MTF0 * np.exp(-(sigma_jitter / sigma_crit)**2)
    mtf = np.clip(mtf_base + rng.normal(0, 0.01, n_out), 0.3, 1.0)

    # ---- 理论预测 ----
    # 闭环跟踪误差: e_track = A * w_cmd / Kp_pos
    # 稳态误差: e_ss = Tf_deg / Kp_pos
    A_az, omega_az = 45.0, 2 * np.pi / 12.0
    A_el, omega_el = 28.0, 2 * np.pi / 9.0
    e_theory_track_az = A_az * omega_az / pid_az["Kp_pos"]
    e_theory_track_el = A_el * omega_el / pid_el["Kp_pos"]
    # 稳态误差: 库仑摩擦 / 等效位置刚度
    Tf_deg_az = deg_az["Tf"] / deg_az["Kt"] * deg_az["R"] * 180 / np.pi
    Tf_deg_el = deg_el["Tf"] / deg_el["Kt"] * deg_el["R"] * 180 / np.pi
    e_theory_ss_az = Tf_deg_az / max(pid_az["Kp_pos"], 0.01)
    e_theory_ss_el = Tf_deg_el / max(pid_el["Kp_pos"], 0.01)

    # ---- 组装数据 ----
    data = np.column_stack([
        np.round(t_out, 3),
        np.round(az_c_out, 5), np.round(az_p_out, 5), np.round(az_e_out, 5),
        np.round(el_c_out, 5), np.round(el_p_out, 5), np.round(el_e_out, 5),
        np.round(np.clip(I_az, 0, 4), 4), np.round(np.clip(I_el, 0, 3), 4),
        np.round(temp, 2), np.round(vib, 5), np.round(mtf, 4),
        np.full(n_out, 0 if degradation == "normal" else 1)
    ])

    summary = {
        "degradation": degradation,
        "hours": DEG_COEF[degradation]["hours"],
        # 电机参数
        "R_az": round(deg_az["R"], 4), "R_el": round(deg_el["R"], 4),
        "Kt_az": round(deg_az["Kt"], 4), "Kt_el": round(deg_el["Kt"], 4),
        "B_az": round(deg_az["B"], 6), "B_el": round(deg_el["B"], 6),
        "Tf_az": round(deg_az["Tf"], 6), "Tf_el": round(deg_el["Tf"], 6),
        # 仿真结果
        "steady_az": round(s_az, 5), "steady_el": round(s_el, 5),
        "steady_max_deg": round(max(s_az, s_el), 5),
        "track_az": round(t_az, 5), "track_el": round(t_el, 5),
        "track_max_deg": round(max(t_az, t_el), 5),
        # 理论预测
        "theory_track_az": round(e_theory_track_az, 5),
        "theory_track_el": round(e_theory_track_el, 5),
        "theory_steady_az": round(e_theory_ss_az, 5),
        "theory_steady_el": round(e_theory_ss_el, 5),
        # 噪声统计
        "noise_sigma_quant_deg": round(sigma_quant, 6),
        "noise_sigma_thermal_deg": round(sigma_thermal, 6),
        "noise_total_deg": round(sigma_sensor, 6),
        # 达标
        "req_deg": 0.1,
        "steady_pass": max(s_az, s_el) <= 0.1,
        "track_pass": max(t_az, t_el) <= 0.1,
        "mtf_avg": round(float(np.mean(mtf)), 4),
    }

    return data, summary


def save_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(FIELDS)
        for row in data:
            w.writerow(list(row[:-1]) + ["normal" if row[-1] == 0 else "degraded"])


if __name__ == "__main__":
    levels = ["normal", "mild", "moderate", "severe"]
    summaries = {}

    for lv in levels:
        print(f"  {lv}...", end=" ", flush=True)
        data, s = generate_servo(duration=60.0, dt=0.001, degradation=lv, seed=42)
        save_csv(data, os.path.join(OUT_DIR, f"servo_{lv}.csv"))
        summaries[lv] = s

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=float)

    labels = {"normal": "正常", "mild": "轻度", "moderate": "中度", "severe": "重度"}
    print(f"\n{'等级':<8} {'时长h':>6} {'R_az':>6} {'Kt_az':>6} {'Tf_az':>7} "
          f"{'T-稳态':>8} {'T-跟踪':>8} {'S-稳态':>8} {'S-跟踪':>8} {'Req':>6}")
    print(f"{'-'*80}")
    for lv in levels:
        s = summaries[lv]
        print(f"{labels[lv]:<8} {s['hours']:>6} {s['R_az']:>6.2f} {s['Kt_az']:>6.3f} {s['Tf_az']:>7.4f} "
              f"{s['theory_steady_az']:>8.5f} {s['theory_track_az']:>8.5f} "
              f"{s['steady_max_deg']:>8.5f} {s['track_max_deg']:>8.5f} "
              f"{'OK' if s['steady_pass'] else '--':>6}")

    sq = SENSOR_RESOLUTION / np.sqrt(12)
    st = np.sqrt(4 * kB * T_AMB * THERMAL_R * BANDWIDTH) * 0.1
    ss = np.sqrt(sq**2 + st**2)
    print(f"\n  噪声模型: 量化σ={sq:.6f}° | 热σ={st:.6f}° | 综合σ={ss:.6f}°")
    print(f"  理论公式: e_track = A*w*J/Kt | e_ss = Tf/(Kp*Kt/R)")
    print(f"  退化规律: R(t)=R0(1+a*t) | Kt(t)=Kt0*exp(-b*t) | B(t)=B0(1+c*t^1.5) | Tf渐近增大")
