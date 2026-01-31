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
    
    simulated_elims = [set() for _ in range(T)]
    active_mask = np.ones(N, dtype=bool)
    
    for t in range(T):
        if not ran[t]: continue
        
        current_active = [i for i in range(N) if active_mask[i] and (t, i) in meta["var_index"]]
        if len(current_active) <= 1: break
        
        judge_scores = np.array([q.get((t, i), 0.0) for i in current_active])
        audience_shares = np.array([V[t, i] for i in current_active])
        
        if rule == 'percent':
            total_scores = judge_scores + audience_shares
        else:
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
            
        if len(E_true[t]) > 0:
            simulated_elims[t].add(eliminated_idx)
            active_mask[eliminated_idx] = False
            
    return simulated_elims

def analyze_problem_2(df, week_judge_cols, max_week):
    print("\n--- Problem 2: Voting Rule Comparison ---")
    target_seasons = [2, 4, 11, 27]
    results = []

    for sid in target_seasons:
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        
        contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
        E = elimination_set(J, ran)
        prob = build_problem_percent(contestants, J, ran, E)
        res, meta, V = solve_percent_inverse(prob)
        
        if V is None: continue
        
        elims_percent = simulate_elimination(meta, V, rule='percent')
        elims_rank = simulate_elimination(meta, V, rule='rank')
        elims_judge_save = simulate_elimination(meta, V, rule='percent', use_judge_save=True)
        
        def calc_accuracy(sim_elims):
            correct = 0
            total = 0
            for t in range(len(E)):
                if len(E[t]) > 0:
                    total += 1
                    if sim_elims[t] == E[t]:
                        correct += 1
            return correct / total if total > 0 else 0

        acc_p = calc_accuracy(elims_percent)
        acc_r = calc_accuracy(elims_rank)
        acc_j = calc_accuracy(elims_judge_save)
        
        results.append({
            "Season": sid,
            "Percent Accuracy": acc_p,
            "Rank Accuracy": acc_r,
            "Judge Save Accuracy": acc_j
        })
        print(f"Season {sid}: Percent Acc={acc_p:.2f}, Rank Acc={acc_r:.2f}, Judge Save Acc={acc_j:.2f}")

        info = df_s[["celebrity_name", "placement"]].drop_duplicates().set_index("celebrity_name")
        contestant_list = meta["contestants"]
        for name in ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]:
            if name in contestant_list:
                actual_rank = info.loc[name, "placement"]
                print(f"  [Controversial] {name}: Actual Placement = {actual_rank}")

    res_df = pd.DataFrame(results)
    res_df.set_index("Season").plot(kind="bar", figsize=(10, 6))
    plt.title("Comparison of Voting Rules Accuracy")
    plt.ylabel("Accuracy Rate")
    plt.tight_layout()
    plt.savefig("problem_2_comparison.png")
    print("Saved problem_2_comparison.png")

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    analyze_problem_2(df, week_judge_cols, max_week)
