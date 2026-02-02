import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import rankdata
from functools import partial
from scipy.optimize import minimize
import re

# 导入你之前的解析函数
# 如果在 VS Code 中运行，请确保这两个函数在 run_reconstruction.py 或本文件中
from run_reconstruction import parse_elimination_week, get_weekly_judge_score

# ==========================================
# 1. 核心求解函数 (支持自定义 Alpha)
# ==========================================
def solve_with_alpha(judge_scores_raw, eliminated_idx, prior_votes, alpha, rule='percent', epsilon=1e-5):
    n = len(judge_scores_raw)
    
    # 数据预处理
    if rule == 'rank':
        J_processed = rankdata(judge_scores_raw, method='min')
    else:
        J_processed = np.array(judge_scores_raw, dtype=float)
        
    J_norm = J_processed / (np.sum(J_processed) if np.sum(J_processed) > 0 else 1.0)
    priors = np.array(prior_votes) / np.sum(prior_votes)

    def objective(v):
        return np.sum((v - priors)**2)

    # 核心约束修改：Score = alpha * J + (1 - alpha) * V
    def constraint_func(v, s_idx, e_idx, J_vals, a_val, eps):
        score_s = a_val * J_vals[s_idx] + (1 - a_val) * v[s_idx]
        score_e = a_val * J_vals[e_idx] + (1 - a_val) * v[e_idx]
        return score_s - score_e - eps

    constraints = [{'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0}]
    for i in range(n):
        if i != eliminated_idx:
            constraints.append({
                'type': 'ineq', 
                'fun': partial(constraint_func, s_idx=i, e_idx=eliminated_idx, J_vals=J_norm, a_val=alpha, eps=epsilon)
            })

    # 运行优化
    res = minimize(objective, priors, method='SLSQP', bounds=[(0.0, 1.0)]*n, constraints=constraints)
    
    # 返回：优化是否成功 (即该 Alpha 下是否存在合理解)
    return res.success

# ==========================================
# 2. 灵敏度扫描逻辑
# ==========================================
def run_alpha_scan(csv_path):
    # 1. 严谨的读取与清洗：处理 BOM 和不可见字符
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    df.columns = [c.encode('ascii', 'ignore').decode('ascii').strip().lower() for c in df.columns]
    
    # 2. 动态识别关键列名
    name_col = next((c for c in ['celebrity_name', 'celebrity', 'celeb_key'] if c in df.columns), None)
    res_col = 'results' if 'results' in df.columns else 'result'
    if not name_col: raise KeyError("Data missing celebrity name column.")

    # 3. 扩大采样范围：选取更具代表性的赛季区间
    # 包含 Rank 早期 (1-2), Percent 时代 (3, 11, 27) 和 Rank 回归时代 (28, 34)
    seasons_to_test = [1, 2, 3, 11, 27, 28, 34] 
    alphas = np.linspace(0.1, 0.9, 9) # 细化步长
    final_results = []

    for a in alphas:
        print(f"Analyzing Sensitivity for Alpha = {a:.1f}...")
        results_at_alpha = []
        
        for s in seasons_to_test:
            df_season = df[df['season'] == s].copy()
            
            # --- 严谨点 1: 严格遵循官方规则切换点 ---
            # 根据附录：S1,2 及 S28+ 使用 Rank；S3-27 使用 Percent
            rule = 'rank' if (s <= 2 or s >= 28) else 'percent'
            
            # --- 严谨点 2: 动态探测本赛季的最大周数 ---
            # 扫描列名中形如 weekX_judge1 的最大 X
            week_cols = [c for c in df.columns if c.startswith('week') and '_judge1' in c]
            max_w = max([int(re.search(r'week(\d+)', c).group(1)) for c in week_cols])
            
            for w in range(1, max_w + 1):
                # 提取当周活跃选手
                active_list = []
                for _, row in df_season.iterrows():
                    j_score = get_weekly_judge_score(row, w)
                    if j_score > 0:
                        active_list.append({
                            'name': row[name_col], 
                            'j_score': j_score, 
                            'elim_w': parse_elimination_week(row[res_col])
                        })
                
                if len(active_list) < 2: continue
                df_w = pd.DataFrame(active_list)
                
                # 确定当周是否有淘汰发生（排除非淘汰周或决赛）
                elim_targets = df_w[df_w['elim_w'] == w]
                if elim_targets.empty: continue
                
                target_idx = elim_targets.index[0]
                j_scores = df_w['j_score'].values
                prior = np.ones(len(j_scores)) / len(j_scores) # 均匀分布作为无偏先验
                
                # 求解优化问题，测试当前权重 alpha 下的逻辑可行性
                is_feasible = solve_with_alpha(j_scores, target_idx, prior, a, rule)
                results_at_alpha.append(is_feasible)
        
        # 计算该 Alpha 权重下的全局一致性
        consistency = np.mean(results_at_alpha) if results_at_alpha else 0
        final_results.append({'alpha': a, 'consistency': consistency})

    return pd.DataFrame(final_results)

# ==========================================
# 3. 运行并绘图
# ==========================================
if __name__ == "__main__":
    # 步骤 A: 运行扫描 (这可能需要几分钟)
    res_df = run_alpha_scan("2026_MCM_Problem_C_Data.csv")
    
    # 步骤 B: 可视化
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x='alpha', y='consistency', marker='o', linewidth=2.5, color='#2c7bb6')
    plt.axvline(0.5, color='#d7191c', linestyle='--', label='Baseline (0.5)')
    
    plt.title('Task 1 Sensitivity Analysis: Impact of Alpha on Model Consistency', fontsize=14)
    plt.xlabel('Judge Weight Factor (Alpha)', fontsize=12)
    plt.ylabel('Feasible Consistency Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 保存结果
    plt.savefig('task1_sensitivity_alpha.png', dpi=300)
    print("灵敏度分析图表已保存至 task1_sensitivity_alpha.png")