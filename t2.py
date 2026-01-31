import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t1_gas import load_data, parse_week_judge_columns, season_matrices, elimination_set, build_problem_percent, solve_percent_inverse

def get_rank_points(scores):
    """
    排名法：分数越高，排名分越高。最高分得 N 分，最低分得 1 分。
    """
    series = pd.Series(scores)
    return series.rank(method='min', ascending=True)

def simulate_elimination(meta, V, rule='percent', use_judge_save=False):
    """
    模拟整个赛季的淘汰过程
    rule: 'percent' (百分比法) or 'rank' (排名法)
    use_judge_save: 是否使用“评委二选一”机制
    """
    T, N = meta["T"], meta["N"]
    ran = meta["ran"]
    q = meta["q"]
    E_true = meta["E"]
    contestants = meta["contestants"]
    
    simulated_elims = [set() for _ in range(T)]
    active_mask = np.ones(N, dtype=bool)
    placements = np.zeros(N, dtype=int)
    elim_order = []
    
    for t in range(T):
        if not ran[t]: continue
        
        current_active = [i for i in range(N) if active_mask[i] and (t, i) in meta["var_index"]]
        if len(current_active) <= 1: break
        
        judge_scores = np.array([q.get((t, i), 0.0) for i in current_active])
        audience_shares = np.array([V[t, i] for i in current_active])
        
        if rule == 'percent':
            total_scores = judge_scores + audience_shares
        else:
            # 排名法
            judge_ranks = get_rank_points(judge_scores).values
            audience_ranks = get_rank_points(audience_shares).values
            total_scores = judge_ranks + audience_ranks
            
        bottom_indices = np.argsort(total_scores)[:2]
        bottom_contestants = [current_active[idx] for idx in bottom_indices]
        
        eliminated_idx = -1
        if use_judge_save and len(bottom_contestants) >= 2:
            s1 = q.get((t, bottom_contestants[0]), 0.0)
            s2 = q.get((t, bottom_contestants[1]), 0.0)
            eliminated_idx = bottom_contestants[0] if s1 <= s2 else bottom_contestants[1]
        else:
            eliminated_idx = bottom_contestants[0]
            
        # 记录模拟的淘汰
        simulated_elims[t].add(eliminated_idx)
        active_mask[eliminated_idx] = False
        elim_order.append(eliminated_idx)
            
    # 计算最终排名
    remaining = [i for i in range(N) if active_mask[i] and any((t, i) in meta["var_index"] for t in range(T))]
    # 假设剩下的按最后一轮分数排名（简化处理）
    # 或者就按淘汰顺序倒序排列
    full_order = elim_order + remaining
    for rank_idx, contestant_idx in enumerate(reversed(full_order)):
        placements[contestant_idx] = rank_idx + 1
        
    return simulated_elims, placements

def calculate_bias(meta, V, simulated_placements):
    """
    计算模拟排名与评委分数、观众投票的相关性，以确定规则偏向。
    """
    N = meta["N"]
    T = meta["T"]
    q = meta["q"]
    
    # 计算每个选手的平均评委分和平均观众分
    avg_judge = []
    avg_audience = []
    ranks = []
    
    for i in range(N):
        j_scores = [q[(t, i)] for t in range(T) if (t, i) in q]
        a_shares = [V[t, i] for t in range(T) if (t, i) in meta["var_index"]]
        
        if len(j_scores) > 0:
            avg_judge.append(np.mean(j_scores))
            avg_audience.append(np.mean(a_shares))
            ranks.append(simulated_placements[i])
            
    if not ranks: return 0, 0
    
    # 使用 Spearman 相关系数（排名相关性）
    # 排名越小（第1名）对应的分数应该越大，所以期望负相关
    from scipy.stats import spearmanr
    corr_judge, _ = spearmanr(ranks, avg_judge)
    corr_audience, _ = spearmanr(ranks, avg_audience)
    
    return corr_judge, corr_audience

def analyze_problem_2(df, week_judge_cols, max_week):
    print("\n--- Problem 2: Voting Rule Comparison & Counterfactual Analysis ---")
    target_seasons = [2, 4, 11, 27]
    results = []
    bias_results = []
    controversial_fate = []

    for sid in target_seasons:
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        
        contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
        E = elimination_set(J, ran)
        prob = build_problem_percent(contestants, J, ran, E)
        res, meta, V = solve_percent_inverse(prob)
        
        if V is None: continue
        
        elims_p, place_p = simulate_elimination(meta, V, rule='percent')
        elims_r, place_r = simulate_elimination(meta, V, rule='rank')
        elims_j, place_j = simulate_elimination(meta, V, rule='percent', use_judge_save=True)
        
        def calc_accuracy(sim_elims):
            correct = 0
            total = 0
            for t in range(len(E)):
                if len(E[t]) > 0:
                    total += 1
                    # 检查模拟淘汰的选手是否在真实淘汰集合中
                    if sim_elims[t].intersection(E[t]):
                        correct += 1
            return correct / total if total > 0 else 0

        acc_p = calc_accuracy(elims_p)
        acc_r = calc_accuracy(elims_r)
        acc_j = calc_accuracy(elims_j)
        
        results.append({
            "Season": sid,
            "Percent Accuracy": acc_p,
            "Rank Accuracy": acc_r,
            "Judge Save Accuracy": acc_j
        })
        
        # 偏差分析
        bj_p, ba_p = calculate_bias(meta, V, place_p)
        bj_r, ba_r = calculate_bias(meta, V, place_r)
        
        bias_results.append({
            "Season": sid,
            "Rule": "Percent",
            "Judge Correlation": bj_p,
            "Audience Correlation": ba_p
        })
        bias_results.append({
            "Season": sid,
            "Rule": "Rank",
            "Judge Correlation": bj_r,
            "Audience Correlation": ba_r
        })

        print(f"Season {sid}: Percent Acc={acc_p:.2f}, Rank Acc={acc_r:.2f}, Judge Save Acc={acc_j:.2f}")
        print(f"  Bias (Percent): Judge={bj_p:.2f}, Audience={ba_p:.2f}")
        print(f"  Bias (Rank):    Judge={bj_r:.2f}, Audience={ba_r:.2f}")

        # 争议选手分析
        info = df_s[["celebrity_name", "placement"]].drop_duplicates().set_index("celebrity_name")
        contestant_list = meta["contestants"]
        for name in ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]:
            if name in contestant_list:
                idx = contestant_list.index(name)
                actual_rank = info.loc[name, "placement"]
                sim_rank_p = place_p[idx]
                sim_rank_r = place_r[idx]
                sim_rank_j = place_j[idx]
                
                controversial_fate.append({
                    "Season": sid,
                    "Name": name,
                    "Actual": actual_rank,
                    "Percent": sim_rank_p,
                    "Rank": sim_rank_r,
                    "JudgeSave": sim_rank_j
                })
                print(f"  [Controversial] {name}: Actual={actual_rank}, Sim_P={sim_rank_p}, Sim_R={sim_rank_r}, Sim_J={sim_rank_j}")

    # 可视化结果
    res_df = pd.DataFrame(results)
    res_df.set_index("Season").plot(kind="bar", figsize=(10, 5))
    plt.title("Accuracy of Different Rules")
    plt.ylabel("Accuracy Rate")
    plt.tight_layout()
    plt.savefig("problem_2_accuracy.png")
    
    bias_df = pd.DataFrame(bias_results)
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bias_df, x="Season", y="Audience Correlation", hue="Rule")
    plt.title("Rule Bias towards Audience Opinion (Correlation)")
    plt.ylabel("Spearman Correlation (Negative is better for rank)")
    plt.tight_layout()
    plt.savefig("problem_2_bias_audience.png")

    fate_df = pd.DataFrame(controversial_fate)
    print("\nControversial Contestants Fate Summary:")
    print(fate_df)
    
    print("\nSaved problem_2_accuracy.png and problem_2_bias_audience.png")

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    analyze_problem_2(df, week_judge_cols, max_week)
