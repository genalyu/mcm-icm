import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.stats import kendalltau
# 确保 solver_strict.py 在同一目录下
from solver_strict import solve_fan_votes_strictly

# ==========================================
# 1. 配置区域
# ==========================================
CSV_PATH = "2026_MCM_Problem_C_Data.csv"

# 关键列名映射 (根据你的CSV文件)
COL_SEASON = 'season'
COL_NAME = 'celebrity_name'
COL_RESULT = 'results'  # 存放 "Eliminated Week 3" 的列

# ==========================================
# 2. 辅助解析函数
# ==========================================
def parse_elimination_week(result_str):
    """
    从 "Eliminated Week 4" 中提取数字 4。
    如果是 "1st Place", "Withdrew" 等，返回 None 或特定标识。
    """
    if not isinstance(result_str, str):
        return None
    
    # 匹配 "Eliminated Week X"
    match = re.search(r'Eliminated Week (\d+)', result_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None # 对于 Winner, Runner-up, Withdrew 等，不视为“被淘汰”

def get_weekly_judge_score(row, week_num):
    """
    计算某人某周的评委总分。
    自动查找 week{w}_judge1_score 到 week{w}_judge4_score
    """
    cols = [f'week{week_num}_judge{i}_score' for i in range(1, 5)]
    scores = []
    for c in cols:
        if c in row.index and pd.notnull(row[c]):
            scores.append(row[c])
    
    if not scores:
        return 0 # 本周没分（可能已经淘汰或缺席）
    return sum(scores)

# ==========================================
# 3. 主逻辑
# ==========================================
def run_full_reconstruction():
    print(f"正在读取数据: {CSV_PATH} ...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("错误：找不到CSV文件！")
        return

    results = []
    total_weeks_processed = 0
    correct_weeks_count = 0
    taus = []

    # 获取所有赛季
    seasons = sorted(df[COL_SEASON].unique())

    print("开始逐周重构 (Solving Inverse Problem)...")
    
    for s in tqdm(seasons):
        # 1. 筛选本赛季数据
        df_season = df[df[COL_SEASON] == s].copy()
        
        # 2. 确定规则 (S1-S10: Rank, S11+: Percent)
        # 这是一个常见的划分，你可以根据题目微调
        rule = 'rank' if s <= 10 else 'percent' 
        
        # 3. 确定本赛季最大周数 (通过列名推断，或者硬编码 12)
        # 我们扫描所有 weekX_judge1 列是否存在
        max_week = 0
        for col in df_season.columns:
            if col.startswith("week") and "judge" in col:
                try:
                    w = int(re.search(r'week(\d+)_', col).group(1))
                    max_week = max(max_week, w)
                except:
                    pass
        
        # 逐周遍历
        for w in range(1, max_week + 1):
            # A. 找出本周参赛的选手 (Active Contestants)
            # 条件：本周有评委分 > 0
            active_data = []
            
            for idx, row in df_season.iterrows():
                j_score = get_weekly_judge_score(row, w)
                if j_score > 0:
                    # 还要检查他是否在这一周之前就已经被淘汰了？
                    # 逻辑上，如果这周有分，说明他还在跳。
                    # 但为了保险，可以检查 parse_elimination_week(row[COL_RESULT]) < w
                    elim_w = parse_elimination_week(row[COL_RESULT])
                    if elim_w is not None and elim_w < w:
                        continue # 理论上不该发生，防止数据错误
                    
                    active_data.append({
                        'name': row[COL_NAME],
                        'j_score': j_score,
                        'elim_week': elim_w,
                        'result_str': row[COL_RESULT]
                    })
            
            if not active_data:
                continue # 本周没数据
            
            # 转为 DataFrame 方便处理
            df_active = pd.DataFrame(active_data)
            n_contestants = len(df_active)
            
            # B. 找出谁本周被淘汰 (Target)
            # 逻辑：result 列里写了 "Eliminated Week {w}" 的人
            eliminated_mask = (df_active['elim_week'] == w)
            eliminated_candidates = df_active[eliminated_mask]
            
            if eliminated_candidates.empty:
                # 本周是非淘汰周 (Non-elimination week) 或 决赛
                # 我们跳过重构，或者假设均匀分布
                continue
            
            # C. 准备 Solver 需要的输入
            names = df_active['name'].values
            j_scores = df_active['j_score'].values
            
            # 获取被淘汰者的 index
            # 如果有双重淘汰，我们取第一个 (Strict Solver目前支持单人约束)
            # 或者取评委分较高的那个 (因为他更难死，约束更紧)
            target_name = eliminated_candidates.iloc[0]['name']
            target_idx = np.where(names == target_name)[0][0]
            
            # Prior: 假设评委分越高，人气越高 (作为初值)
            # 加一点随机扰动，模拟 "Unknown Secret"
            priors = j_scores + np.random.normal(0, 2, size=n_contestants)
            priors = np.maximum(priors, 0.1) # 保证非负
            
            # D. 调用 Solver
            v_reconstructed = solve_fan_votes_strictly(
                judge_scores_raw=j_scores,
                eliminated_idx=target_idx,
                prior_votes=priors,
                rule=rule
            )
            
            # E. 验证与记录
            total_weeks_processed += 1
            
            # 简单的验证逻辑
            if rule == 'percent':
                j_share = j_scores / np.sum(j_scores)
                # v_reconstructed 已经是归一化的
                total_score = 0.5 * j_share + 0.5 * v_reconstructed
            else:
                from scipy.stats import rankdata
                r_j = rankdata(j_scores, method='min')
                # 假设 v 也要转 rank
                r_v = rankdata(v_reconstructed, method='min')
                total_score = r_j + r_v
            
            # 检查被淘汰者是否在安全区外
            # (只要比幸存者的最低分低，或者在 Rank 制下处于底部)
            # 严格验证：Score(Elim) < Min(Score(Survivors))
            survivor_indices = [i for i in range(n_contestants) if i != target_idx]
            
            if len(survivor_indices) > 0:
                s_elim = total_score[target_idx]
                s_min_survivor = np.min(total_score[survivor_indices])
                
                # 允许微小误差 (Rank制下同分可能需要更复杂的规则，这里简化判断)
                if s_elim <= s_min_survivor + 1e-5:
                    correct_weeks_count += 1
                else:
                    # 失败案例 (可取消注释查看)
                    # print(f"Mismatch S{s} W{w}: {target_name} (Score {s_elim:.3f}) vs MinSurvivor ({s_min_survivor:.3f})")
                    pass
            else:
                correct_weeks_count += 1 # 只剩一个人(决赛)，不算失败
            
            # 计算 Tau (和 Prior 的相关性)
            tau, _ = kendalltau(priors, v_reconstructed)
            if not np.isnan(tau):
                taus.append(tau)
            
            # 保存结果
            for i, nm in enumerate(names):
                results.append({
                    'Season': s,
                    'Week': w,
                    'Contestant': nm,
                    'Judge_Score_Raw': j_scores[i],
                    'Reconstructed_Vote_Share': v_reconstructed[i],
                    'Result_Status': df_active.iloc[i]['result_str'],
                    'Is_Eliminated': (i == target_idx)
                })

    # ==========================================
    # 4. 输出报告
    # ==========================================
    accuracy = correct_weeks_count / total_weeks_processed if total_weeks_processed > 0 else 0
    avg_tau = np.mean(taus) if taus else 0

    print("\n" + "="*40)
    print(f"Task 1 重构结果报告 (Wide Format Adapted)")
    print("="*40)
    print(f"总处理淘汰周数: {total_weeks_processed}")
    print(f"成功复现周数:   {correct_weeks_count}")
    print(f"准确率 (Consistency): {accuracy:.2%}")
    print(f"平均 Kendall's Tau: {avg_tau:.4f}")
    print("-" * 40)
    
    # 保存 CSV
    df_out = pd.DataFrame(results)
    # 填充空缺 (非参赛周) 没必要，这只是稀疏记录
    output_file = "task1_reconstructed_votes.csv"
    df_out.to_csv(output_file, index=False)
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    run_full_reconstruction()