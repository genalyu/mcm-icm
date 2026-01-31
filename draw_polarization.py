import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 引入 Task 1 的求解器
try:
    from t1_gas import load_data, parse_week_judge_columns, season_matrices, elimination_set, build_problem_percent, solve_percent_inverse
except ImportError:
    print("错误：未找到 t1_gas.py，请确保该文件在同一目录下。")
    exit()

# ==========================================
# 1. 数据收集 (保持不变)
# ==========================================
def collect_plot_data(df, week_judge_cols, max_week):
    plot_data = []
    all_seasons = sorted(df['season'].unique())
    highlight_names = ["Jerry Rice", "Bobby Bones", "Billy Ray Cyrus", "Bristol Palin", "Master P"] # 把 Master P 也加进高亮名单

    print("正在计算所有赛季的潜在投票分布 (V)...")
    for sid in tqdm(all_seasons):
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        try:
            contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
            E = elimination_set(J, ran)
            prob = build_problem_percent(contestants, J, ran, E)
            res, meta, V = solve_percent_inverse(prob)
        except:
            continue
        if V is None: continue
        T, N = meta["T"], meta["N"]
        q = meta["q"]
        for t in range(T):
            if not ran[t]: continue
            active_indices = [i for i in range(N) if (t, i) in meta["var_index"]]
            if not active_indices: continue
            raw_scores = [q.get((t, i), 0.0) for i in active_indices]
            total_judge_points = sum(raw_scores)
            if total_judge_points == 0: continue
            for idx, i in enumerate(active_indices):
                name = contestants[i]
                raw_j = raw_scores[idx]
                j_share = raw_j / total_judge_points
                v_share = V[t, i]
                # 只要在高亮名单里，就标记为 Highlight
                is_highlight = name in highlight_names
                plot_data.append({
                    "Name": name,
                    "Season": sid,
                    "Week": t + 1,
                    "Judge_Share": j_share,
                    "Vote_Share": v_share,
                    "Type": "Highlight" if is_highlight else "Normal"
                })
    return pd.DataFrame(plot_data)

# ==========================================
# 2. 绘图 (最终进化版)
# ==========================================
def draw_polarization_plot(df_plot):
    # 设置风格
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 11)) # 稍微大一点，给文字留空间
    
    # --- A. 画背景云团 (优化版：更醒目) ---
    normal_data = df_plot[df_plot["Type"] == "Normal"]
    sns.scatterplot(
        data=normal_data, x="Judge_Share", y="Vote_Share",
        color="#7f8c8d",  # 深灰色，更有质感
        alpha=0.5,        # 透明度调高，不再是“淡淡的”
        s=40,             # 点稍微大一点
        ax=ax, 
        label="Regular Performances", 
        edgecolor='white', # 加个白边，像玻璃珠一样清晰
        linewidth=0.5
    )
    
    # --- B. 画高亮人物 ---
    highlight_data = df_plot[df_plot["Type"] == "Highlight"]
    
    # 颜色盘 (Master P 既然是特例，给他一个特殊的深紫色)
    palette = {
        "Jerry Rice": "#e74c3c",       # 红
        "Bobby Bones": "#e67e22",      # 橙
        "Billy Ray Cyrus": "#9b59b6",  # 紫
        "Bristol Palin": "#3498db",    # 蓝
        "Master P": "#2c3e50"          # 深蓝黑 (大佬色)
    }
    
    sns.scatterplot(
        data=highlight_data, x="Judge_Share", y="Vote_Share", hue="Name", palette=palette,
        style="Name", s=130, alpha=1.0, ax=ax, edgecolor='white', linewidth=1.5, zorder=10
    )
    
    # --- C. 画对角线 y=x ---
    limit = max(df_plot["Judge_Share"].max(), df_plot["Vote_Share"].max()) + 0.05
    ax.plot([0, limit], [0, limit], ls="--", c="#2c3e50", lw=2, alpha=0.8, label="Perfect Alignment (y=x)")
    
    # --- D. 标注 "Jerry Rice Zone" ---
    ax.text(0.015, limit * 0.95, 
            "The 'Jerry Rice Zone'\n(High Popularity, Low Skill)\nExtreme Polarization", 
            fontsize=13, fontweight='bold', color='#c0392b', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="#fce4ec", ec="#c0392b", alpha=0.9, lw=2))
            
    # --- E. 细节美化 ---
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal')
    ax.set_title("The Polarization Scatter Plot:\nDecoupling Technical Merit from Popularity", 
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel("Normalized Judge Score ($\~{J}_{i,w}$)", fontsize=15, fontweight='bold')
    ax.set_ylabel("Reconstructed Audience Vote Share ($\hat{V}_{i,w}$)", fontsize=15, fontweight='bold')
    ax.legend(title="Contestant", loc='lower right', frameon=True, fancybox=True, framealpha=0.9, fontsize=11)
    
    # ==========================================
    # F. 双重标注 (核心修改)
    # ==========================================
    
    # 计算 Diff 用来找极值
    df_plot['Diff'] = df_plot['Vote_Share'] - df_plot['Judge_Share']

    # --- 标注 1: Master P (全场最大偏差) ---
    max_row = df_plot.loc[df_plot['Diff'].idxmax()] # 自动找到 Master P
    
    ax.annotate(f"The Statistical Extreme\n({max_row['Name']}, S{max_row['Season']})",
                xy=(max_row['Judge_Share'], max_row['Vote_Share']),
                # 文本放在点的【右上方】
                xytext=(max_row['Judge_Share'] + 0.05, max_row['Vote_Share'] + 0.2), # 往右上方移
                # xytext=(max_row['Judge_Share'] + 0.1, max_row['Vote_Share'] + 0.05), 
                # 箭头带弧度 (rad=-0.2)
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black', lw=2),
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="black"),
                zorder=20)

    # --- 标注 2: Jerry Rice (题目关注点) ---
    # 找到 Jerry Rice 偏差最大的一周
    jerry_data = df_plot[df_plot['Name'] == "Jerry Rice"]
    if not jerry_data.empty:
        jerry_max = jerry_data.loc[jerry_data['Diff'].idxmax()]
        
        ax.annotate(f"The Focal Case\n(Jerry Rice, S{jerry_max['Season']})",
                    xy=(jerry_max['Judge_Share'], jerry_max['Vote_Share']),
                    # 文本放在 Master P 的【下方】，避免重叠
                    # Jerry 的点通常比 Master P 低一点，所以文本往右放，稍微往下一点
                    xytext=(jerry_max['Judge_Share'] + 0.12, jerry_max['Vote_Share'] - 0.08), 
                    # 箭头弧度相反 (rad=0.3)，形成视觉上的错落感
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='#c0392b', lw=2),
                    fontsize=11, fontweight='bold', color='#c0392b',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#c0392b"),
                    zorder=20)

    plt.tight_layout()
    plt.savefig("polarization_scatter_final.png", dpi=300)
    print("\n[Output] 最终版图表已保存至: polarization_scatter_final.png")
    plt.show()

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    try:
        df = load_data(CSV_PATH)
        week_judge_cols, weeks = parse_week_judge_columns(df)
        max_week = max(weeks)
        print("--- Step 1: Collecting Data Points ---")
        df_plot = collect_plot_data(df, week_judge_cols, max_week)
        print(f"--- Step 2: Drawing Plot ---")
        draw_polarization_plot(df_plot)
    except FileNotFoundError:
        print("错误：找不到数据文件 2026_MCM_Problem_C_Data.csv")