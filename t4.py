import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from t1_nn import load_data, parse_week_judge_columns, season_matrices, elimination_set, build_problem_percent, solve_percent_inverse

def calculate_dynamic_weight(audience_shares):
    """
    根据观众投票的熵（Entropy）来决定评委权重 alpha。
    熵越高（投票越分散/不确定），增加评委权重以保证专业性。
    熵越低（观众意见越一致），减少评委权重以尊重民意。
    """
    # 归一化熵 (0 到 1)
    n = len(audience_shares)
    if n <= 1: return 0.5
    
    ent = entropy(audience_shares) / np.log(n)
    
    # 映射到 [0.3, 0.7] 的权重范围
    # 如果 ent = 1 (完全分散), alpha = 0.7
    # 如果 ent = 0 (完全集中), alpha = 0.3
    alpha = 0.3 + 0.4 * ent
    return alpha

def simulate_with_rule(meta, V, rule_type='fixed'):
    """
    模拟淘汰过程
    rule_type: 'fixed' (50/50) or 'dynamic' (Dynamic Weighting)
    """
    T, N = meta["T"], meta["N"]
    ran = meta["ran"]
    q = meta["q"]
    E_true = meta["E"]
    
    simulated_elims = [set() for _ in range(T)]
    active_mask = np.ones(N, dtype=bool)
    weights_used = []
    
    for t in range(T):
        if not ran[t]: continue
        
        current_active = [i for i in range(N) if active_mask[i] and (t, i) in meta["var_index"]]
        if len(current_active) <= 1: break
        
        judge_scores = np.array([q.get((t, i), 0.0) for i in current_active])
        audience_shares = np.array([V[t, i] for i in current_active])
        
        # 确保总和为1
        if audience_shares.sum() > 0:
            audience_shares = audience_shares / audience_shares.sum()
        
        if rule_type == 'fixed':
            alpha = 0.5
        else:
            alpha = calculate_dynamic_weight(audience_shares)
        
        weights_used.append(alpha)
        total_scores = alpha * judge_scores + (1 - alpha) * audience_shares
        
        eliminated_idx = current_active[np.argmin(total_scores)]
        
        if len(E_true[t]) > 0:
            simulated_elims[t].add(eliminated_idx)
            active_mask[eliminated_idx] = False
            
    return simulated_elims, weights_used

def analyze_problem_4(df, week_judge_cols, max_week):
    print("\n--- Problem 4: Dynamic Voting System Design ---")
    
    # 选择代表性赛季进行模拟：第 27 季（高争议）和第 28 季（相对正常）
    target_seasons = [27, 28]
    comparison_results = []

    for sid in target_seasons:
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        
        contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
        E = elimination_set(J, ran)
        prob = build_problem_percent(contestants, J, ran, E)
        res, meta, V = solve_percent_inverse(prob)
        
        if V is None: continue
        
        # 1. 模拟固定权重 (50/50)
        elims_fixed, _ = simulate_with_rule(meta, V, rule_type='fixed')
        # 2. 模拟动态权重
        elims_dynamic, weights = simulate_with_rule(meta, V, rule_type='dynamic')
        
        # 计算“极端事件”频率：即评委给最高分但由于规则被淘汰，或者评委给最低分但夺冠
        # 这里简化为：动态规则是否改变了最终前三名的构成
        def get_finalists(elims, n=3):
            eliminated_order = []
            for t_set in elims:
                eliminated_order.extend(list(t_set))
            # 剩余的就是决赛选手
            survivors = [i for i in range(meta["N"]) if i not in eliminated_order]
            return set(survivors)

        fixed_finalists = get_finalists(elims_fixed)
        dyn_finalists = get_finalists(elims_dynamic)
        
        print(f"Season {sid}: Avg Dynamic Alpha = {np.mean(weights):.3f}")
        print(f"  Fixed Finalists: {[meta['contestants'][i] for i in fixed_finalists]}")
        print(f"  Dynamic Finalists: {[meta['contestants'][i] for i in dyn_finalists]}")
        
        comparison_results.append({
            "Season": sid,
            "Avg Alpha": np.mean(weights),
            "Weight Std": np.std(weights)
        })

    # 可视化动态权重的变化
    plt.figure(figsize=(10, 5))
    plt.bar([f"S{r['Season']}" for r in comparison_results], [r['Avg Alpha'] for r in comparison_results], yerr=[r['Weight Std'] for r in comparison_results], capsize=5)
    plt.axhline(0.5, color='r', linestyle='--', label='Current Fixed Weight (0.5)')
    plt.title("Dynamic Weighting: Average Judge Weight (Alpha) by Season")
    plt.ylabel("Judge Weight (Alpha)")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("problem_4_dynamic_weight.png")
    print("Saved problem_4_dynamic_weight.png")

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    analyze_problem_4(df, week_judge_cols, max_week)
