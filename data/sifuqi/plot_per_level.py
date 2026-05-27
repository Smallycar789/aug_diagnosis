"""
每个退化等级一张大图, 6个子图覆盖全部参数
输出: normal.png, mild.png, moderate.png, severe.png
"""
import numpy as np, json, os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
levels = ["normal", "mild", "moderate", "severe"]
prefix = {"normal": "1", "mild": "2", "moderate": "3", "severe": "4"}
labels_cn = {"normal": "正常工况", "mild": "轻度退化", "moderate": "中度退化", "severe": "重度退化"}
colors = {"normal": "#3c763d", "mild": "#8a6d3b", "moderate": "#d9534f", "severe": "#a94442"}


def read_csv_float(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            rows.append([float(x) for x in parts[:12]])
    return np.array(rows)


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
except ImportError:
    print("matplotlib 不可用, 请先 pip install matplotlib --upgrade")
    exit(1)


def plot_one_level(lv, data, s):
    label = labels_cn[lv]
    color = colors[lv]
    dt = data[1, 0] - data[0, 0] if len(data) > 1 else 0.01
    t, az_cmd, az_act, az_err = data[:,0], data[:,1], data[:,2], data[:,3]
    el_cmd, el_act, el_err = data[:,4], data[:,5], data[:,6]
    I_az, I_el = data[:,7], data[:,8]
    temp, vib, mtf_q = data[:,9], data[:,10], data[:,11]

    n15 = min(len(t), int(15.0 / dt))
    n10 = min(len(t), int(10.0 / dt))

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # ---- [0,0] 指令 vs 实际 (方位+俯仰, 前10s) ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t[:n10], az_cmd[:n10], 'gray', lw=1.5, ls='--', alpha=0.8)
    ax.plot(t[:n10], az_act[:n10], color, lw=1.0)
    ax.plot(t[:n10], el_cmd[:n10], 'silver', lw=1.2, ls=':', alpha=0.7)
    ax.plot(t[:n10], el_act[:n10], '#3b5998', lw=0.8)
    ax.set_ylabel("角度 (deg)"); ax.legend(['方位指令','方位实际','俯仰指令','俯仰实际'], fontsize=7, ncol=2)
    ax.set_ylim(-55, 55); ax.grid(True, alpha=0.3)
    ax.set_title(f"{label} — 指令 vs 实际", fontsize=12, fontweight='bold')

    # ---- [0,1] 误差时序 (方位+俯仰, 全时域) ----
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, az_err, color, lw=0.3, alpha=0.7, label='方位误差')
    ax.plot(t, el_err, '#3b5998', lw=0.3, alpha=0.5, label='俯仰误差')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel("误差 (deg)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    max_e = max(np.abs(az_err).max(), np.abs(el_err).max()) * 1.15
    ax.set_ylim(-max_e, max_e)
    ax.text(0.99, 0.95, f"跟踪误差max: 方位{np.abs(az_err).max():.3f}deg | 俯仰{np.abs(el_err).max():.3f}deg",
            transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_title(f"跟踪误差 — 全时域", fontsize=12, fontweight='bold')

    # ---- [1,0] 电流 (全时域) ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, I_az, color, lw=0.4, alpha=0.7, label='方位')
    ax.plot(t, I_el, '#3b5998', lw=0.4, alpha=0.5, label='俯仰')
    ax.set_ylabel("电流 (A)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.95,
        f"方位: mean={np.mean(I_az):.3f}  max={np.max(I_az):.2f}  RMS={np.sqrt(np.mean(I_az**2)):.3f}A\n"
        f"俯仰: mean={np.mean(I_el):.3f}  max={np.max(I_el):.2f}  RMS={np.sqrt(np.mean(I_el**2)):.3f}A",
        transform=ax.transAxes, ha='right', va='top', fontsize=7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_title(f"电机电流", fontsize=12, fontweight='bold')

    # ---- [1,1] 温度 ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, temp, color, lw=0.8)
    ax.fill_between(t, np.min(temp), temp, alpha=0.06, color=color)
    ax.set_ylabel("温度 (°C)"); ax.grid(True, alpha=0.2)
    ax.text(0.99, 0.95, f"温度范围: {np.min(temp):.1f} ~ {np.max(temp):.1f}°C",
            transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"温度", fontsize=12, fontweight='bold')

    # ---- [2,0] 振动 ----
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, vib, color, lw=0.3, alpha=0.7)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("振动 (g)"); ax.grid(True, alpha=0.2)
    ax.text(0.99, 0.95, f"振动RMS={np.sqrt(np.mean(vib**2)):.5f}g | max={np.max(vib):.4f}g",
            transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"振动信号", fontsize=12, fontweight='bold')

    # ---- [2,1] 指标汇总面板 ----
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    # 从 summary 提取数据
    s_az_st = s.get('steady_az', 0)
    s_el_st = s.get('steady_el', 0)
    s_az_tr = s.get('track_az', 0)
    s_el_tr = s.get('track_el', 0)
    r_az = s.get('R_az', 0)
    kt_az = s.get('Kt_az', 0)
    tf_az = s.get('Tf_az', 0)
    r_el = s.get('R_el', 0)
    kt_el = s.get('Kt_el', 0)
    tf_el = s.get('Tf_el', 0)

    table_lines = [
        f"{'指标':<22} {'方位轴':>14} {'俯仰轴':>14}",
        f"{'-'*50}",
        f"{'稳态误差 (deg)':<22} {s_az_st:>14.4f} {s_el_st:>14.4f}",
        f"{'跟踪误差 (deg)':<22} {s_az_tr:>14.4f} {s_el_tr:>14.4f}",
        f"{'误差RMS (deg)':<22} {np.sqrt(np.mean(az_err**2)):>14.4f} {np.sqrt(np.mean(el_err**2)):>14.4f}",
        f"",
        f"{'电阻 R (Ohm)':<22} {r_az:>14.2f} {r_el:>14.2f}",
        f"{'力矩常数 Kt (Nm/A)':<22} {kt_az:>14.3f} {kt_el:>14.3f}",
        f"{'摩擦 Tf (Nm)':<22} {tf_az:>14.4f} {tf_el:>14.4f}",
        f"",
        f"{'电流均值 (A)':<22} {np.mean(I_az):>14.3f} {np.mean(I_el):>14.3f}",
        f"{'电流max (A)':<22} {np.max(I_az):>14.2f} {np.max(I_el):>14.2f}",
        f"{'温度范围 (C)':<22} {f'{np.min(temp):.0f}~{np.max(temp):.0f}':>14}",
        f"{'振动RMS (g)':<22} {np.sqrt(np.mean(vib**2)):>14.5f}",
        f"{'MTF 平均':<22} {np.mean(mtf_q):>14.4f}",
    ]
    table_text = "\n".join(table_lines)
    ax.text(0.5, 0.5, table_text, transform=ax.transAxes, ha='center', va='center',
            fontsize=8, fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='#fafbfc', alpha=0.95, edgecolor='#c8ccd4'))
    ax.set_title(f"指标总览", fontsize=12, fontweight='bold')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, hspace=0.4, wspace=0.3)
    fig.savefig(os.path.join(OUT_DIR, f"{prefix[lv]}{lv}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {prefix[lv]}{lv}.png")


if __name__ == "__main__":
    with open(os.path.join(OUT_DIR, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    for lv in levels:
        data = read_csv_float(os.path.join(OUT_DIR, f"servo_{lv}.csv"))
        plot_one_level(lv, data, summary[lv])

    print(f"\n4 张图已生成: 1normal.png, 2mild.png, 3moderate.png, 4severe.png")
