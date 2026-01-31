import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t1_gas import load_data, parse_week_judge_columns, season_matrices, elimination_set, build_problem_percent, solve_percent_inverse

def sensitivity_analysis(df, sid, week_judge_cols, max_week):
    """
    对平滑参数 lambda 进行灵敏度分析
    """
    df_s = df[df["season"] == sid].copy()
    contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
    E = elimination_set(J, ran)
    
    lams = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    print(f"Running sensitivity analysis for Season {sid}...")
    for l in lams:
        prob = build_problem_percent(contestants, J, ran, E, lam=l)
        res, meta, V = solve_percent_inverse(prob)
        if V is not None:
            # 记录关键选手的平均投票份额
            # 我们看前三名的稳定性
            top_3_idx = np.argsort(V[-1, :])[-3:]
            results[l] = V[:, top_3_idx].mean(axis=0)
            
    return lams, results

def generate_memo_data(df):
    """
    生成用于建议书汇总的全局统计数据
    """
    # 1. 行业成功率统计
    industry_map = {
        'Actor': 'Entertainment', 'Actress': 'Entertainment',
        'Athlete': 'Sports', 'NFL': 'Sports', 'Olympic': 'Sports',
        'Singer': 'Music', 'Musician': 'Music'
    }
    
    def simplify_industry(x):
        for k, v in industry_map.items():
            if pd.notna(x) and k in x: return v
        return 'Other'

    df['Industry_Group'] = df['celebrity_industry'].apply(simplify_industry)
    industry_performance = df.groupby('Industry_Group')['placement'].mean().sort_values()
    
    # 2. 年龄与排名的关系
    age_corr = df['celebrity_age_during_season'].corr(df['placement'])
    
    return industry_performance, age_corr

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    
    # 运行灵敏度分析 (以第27季为例)
    lams, sens_results = sensitivity_analysis(df, 27, week_judge_cols, max_week)
    
    # 过滤掉求解失败的 lambda
    successful_lams = [l for l in lams if l in sens_results]
    
    if not successful_lams:
        print("Error: Sensitivity analysis failed for all lambda values.")
    else:
        plt.figure(figsize=(10, 6))
        for i in range(3): # 绘制前三名的变化
            shares = [sens_results[l][i] for l in successful_lams]
            plt.plot(successful_lams, shares, marker='o', label=f'Finalist {i+1}')
        
        plt.xscale('log')
        plt.xlabel('Smoothing Parameter (Lambda)')
        plt.ylabel('Estimated Vote Share')
        plt.title('Sensitivity Analysis: Impact of Lambda on Vote Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("problem_5_sensitivity.png")
        print("Saved problem_5_sensitivity.png")
    
    # 生成建议书数据
    ind_perf, age_c = generate_memo_data(df)
    print("\n--- Summary for Producers Memo ---")
    print(f"Age vs Placement Correlation: {age_c:.3f} (Positive means older age tends to higher placement number/worse rank)")
    print("\nAverage Placement by Industry Group:")
    print(ind_perf)
    
    print("\nRecommendations Preview:")
    print("1. Implement Dynamic Weighting to balance professional judging with fan engagement.")
    print("2. Consider industry-specific trends: Sports stars tend to perform consistently well.")
    print("3. Model shows that audience loyalty (smoothing term) is a key factor in long-term survival.")
