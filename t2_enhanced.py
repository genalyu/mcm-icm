import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 如果没有装 tqdm，可以用 pip install tqdm 安装，或者删掉相关修饰

# 假设 t1_gas 就在同级目录下
from t1_gas import load_data, parse_week_judge_columns, season_matrices, elimination_set, build_problem_percent, solve_percent_inverse

# ==========================================
# 核心逻辑修改区
# ==========================================

def get_rank_points(scores):
    """
    排名法：分数越高，排名分越高。最高分得 N 分，最低分得 1 分。
    """
    series = pd.Series(scores)
    # method='min' 意味着如果有并列，取最小排名（保守估计）
    return series.rank(method='min', ascending=True)

def simulate_elimination(meta, V, rule='percent', use_judge_save=False):
    """
    模拟整个赛季的淘汰过程 (升级版)
    :param rule: 'percent' (百分比法) or 'rank' (排名法)
    :param use_judge_save: 是否启用评委拯救机制
    """
    T, N = meta["T"], meta["N"]
    ran = meta["ran"]
    q = meta["q"]
    E_true = meta["E"]
    
    simulated_elims = [set() for _ in range(T)]
    active_mask = np.ones(N, dtype=bool)
    
    # 记录每个人的淘汰周次 (用于微观分析)
    # 初始化为 -1 (代表存活到最后/夺冠)
    elimination_week_record = {i: -1 for i in range(N)}
    
    for t in range(T):
        if not ran[t]: continue
        
        # 获取当前存活且本周有比赛的选手
        current_active = [i for i in range(N) if active_mask[i] and (t, i) in meta["var_index"]]
        if len(current_active) <= 1: break # 只剩冠军了，停止
        
        judge_scores = np.array([q.get((t, i), 0.0) for i in current_active])
        audience_shares = np.array([V[t, i] for i in current_active])
        
        # --- 核心修改 1: 尺度修复 ---
        if rule == 'percent':
            # 必须先把评委分归一化成百分比，才能和观众百分比相加
            # 假设权重为 50/50 (α=0.5)
            j_sum = judge_scores.sum()
            if j_sum > 0:
                judge_shares = judge_scores / j_sum
            else:
                judge_shares = np.zeros_like(judge_scores)
            
            # Total Score = 0.5 * Judge% + 0.5 * Fan%
            total_scores = 0.5 * judge_shares + 0.5 * audience_shares
            
        else: # rule == 'rank'
            # 排名法：直接算排名的数值
            judge_ranks = get_rank_points(judge_scores).values
            audience_ranks = get_rank_points(audience_shares).values
            total_scores = judge_ranks + audience_ranks
            
        # 找出倒数两名 (Bottom 2)
        bottom_indices = np.argsort(total_scores)[:2]
        bottom_contestants = [current_active[idx] for idx in bottom_indices]
        
        eliminated_idx = -1
        
        # --- 核心修改 2: 评委拯救机制 (Judge Save) ---
        if use_judge_save and len(bottom_contestants) >= 2:
            # 比较两人的评委原始分
            c1 = bottom_contestants[0]
            c2 = bottom_contestants[1]
            s1 = q.get((t, c1), 0.0)
            s2 = q.get((t, c2), 0.0)
            
            # 评委通常会救分数高的那个，淘汰分数低的那个
            # 如果 s1 < s2，淘汰 c1；否则淘汰 c2
            eliminated_idx = c1 if s1 <= s2 else c2
        else:
            # 没有拯救机制，直接淘汰总分最低的
            eliminated_idx = bottom_contestants[0]
            
        # 记录淘汰结果 (注意：这里我们模拟的是“只淘汰一个人”)
        # 为了对比准确率，我们要看历史数据这周是不是真的有人被淘汰
        if len(E_true[t]) > 0:
            simulated_elims[t].add(eliminated_idx)
            active_mask[eliminated_idx] = False
            elimination_week_record[eliminated_idx] = t # 记录他是第几周死的
            
    return simulated_elims, elimination_week_record

# ==========================================
# 分析主程序
# ==========================================

def analyze_problem_2(df, week_judge_cols, max_week):
    print("\n--- Problem 2: Enhanced Comparative Analysis ---")
    
    # 1. 想要关注的争议人物 (微观分析名单)
    watch_list = {
        "Jerry Rice": 2,       # 名字: 赛季
        "Billy Ray Cyrus": 4,
        "Bristol Palin": 11,
        "Bobby Bones": 27
    }
    
    # 存储微观分析结果
    micro_results = []
    
    # 存储宏观分析结果 (准确率)
    macro_results = []

    # 遍历所有赛季 (或者你可以只跑 specific seasons, 但建议跑全量)
    all_seasons = sorted(df['season'].unique())
    
    # 使用 tqdm 显示进度条，因为跑全部赛季可能要几分钟
    for sid in tqdm(all_seasons, desc="Simulating Seasons"):
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        
        # 1. 求解逆问题 (Task 1) 得到 V
        contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
        E = elimination_set(J, ran)
        prob = build_problem_percent(contestants, J, ran, E)
        res, meta, V = solve_percent_inverse(prob)
        
        if V is None: 
            print(f"Season {sid}: Optimization failed, skipping.")
            continue
        
        # 2. 运行三种模拟
        # 模拟 A: 百分比制 (Percent)
        elims_p, weeks_p = simulate_elimination(meta, V, rule='percent', use_judge_save=False)
        # 模拟 B: 排名制 (Rank)
        elims_r, weeks_r = simulate_elimination(meta, V, rule='rank', use_judge_save=False)
        # 模拟 C: 评委拯救 (Judge Save + Percent)
        elims_j, weeks_j = simulate_elimination(meta, V, rule='percent', use_judge_save=True)
        
        # 3. 计算准确率 (Macro)
        def calc_accuracy(sim_elims):
            correct = 0
            total = 0
            for t in range(len(E)):
                if len(E[t]) > 0: # 这周历史上有淘汰
                    total += 1
                    # 只要模拟的淘汰者 在 真实淘汰名单里 就算对 (处理双淘汰情况)
                    if sim_elims[t].intersection(E[t]): 
                        correct += 1
            return correct / total if total > 0 else 0

        acc_p = calc_accuracy(elims_p)
        acc_r = calc_accuracy(elims_r)
        acc_j = calc_accuracy(elims_j)
        
        macro_results.append({
            "Season": sid,
            "Percent Accuracy": acc_p,
            "Rank Accuracy": acc_r,
            "Judge Save Accuracy": acc_j
        })
        
        # 4. 微观追踪 (Micro)
        # 检查这个赛季有没有我们在意的人
        for name, target_sid in watch_list.items():
            if sid == target_sid and name in meta["contestants"]:
                # 获取该选手的 Index
                idx = meta["contestants"].index(name)
                
                # 获取他在三种规则下的存活周数
                # 如果是 -1，说明活到了最后(Finals)；否则是具体的周数
                w_p = weeks_p[idx]
                w_r = weeks_r[idx]
                w_j = weeks_j[idx]
                
                # 获取真实名次 (从原始数据)
                actual_rank = df_s[df_s['celebrity_name'] == name]['placement'].iloc[0]
                
                micro_results.append({
                    "Name": name,
                    "Season": sid,
                    "Actual Placement": actual_rank,
                    "Eliminated Week (Rank Rule)": "Survived" if w_r == -1 else f"Week {w_r}",
                    "Eliminated Week (Percent Rule)": "Survived" if w_p == -1 else f"Week {w_p}",
                    "Eliminated Week (Judge Save)": "Survived" if w_j == -1 else f"Week {w_j}"
                })

    # ==========================================
    # 输出结果
    # ==========================================

    # 1. 绘制宏观准确率对比图
    res_df = pd.DataFrame(macro_results)
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    # 融合数据以便绘图
    melted_df = res_df.melt(id_vars="Season", var_name="Method", value_name="Accuracy")
    
    # sns.barplot(data=melted_df, x="Season", y="Accuracy", hue="Method", palette="viridis")

    palette_dict = {
    "Percent Accuracy": "#1f77b4",     # 保持原来的蓝色
    "Rank Accuracy": "#ffcc00",        # 麦黄色
    "Judge Save Accuracy": "#2ca02c"   # 保持原来的绿色
    }
    sns.barplot(data=melted_df, x="Season", y="Accuracy", hue="Method", palette=palette_dict)

    plt.title("Macro-Analysis: Accuracy of Voting Mechanisms Across Seasons", fontsize=16)
    plt.ylabel("Prediction Accuracy (Match with History)", fontsize=12)
    plt.xlabel("Season", fontsize=12)
    plt.legend(title="Simulation Rule")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("task2_macro_comparison.png", dpi=300)
    print("\n[Output] Graph saved to 'task2_macro_comparison.png'")

    # 2. 打印微观分析表 (重点！)
    micro_df = pd.DataFrame(micro_results)
    print("\n=== Micro-Analysis: The 'Jerry Rice' Effect Counterfactuals ===")
    print(micro_df.to_string(index=False))
    
    # 保存微观表为 CSV 方便你贴到论文里
    micro_df.to_csv("task2_micro_analysis.csv", index=False)
    print("[Output] Table saved to 'task2_micro_analysis.csv'")

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    try:
        df = load_data(CSV_PATH)
        week_judge_cols, weeks = parse_week_judge_columns(df)
        max_week = max(weeks)
        analyze_problem_2(df, week_judge_cols, max_week)
    except FileNotFoundError:
        print("错误：找不到数据文件。请确保 '2026_MCM_Problem_C_Data.csv' 在同一目录下。")
    except ImportError:
        print("错误：找不到 't1_gas.py'。请确保队友的 Task 1 代码在同一目录下。")