import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")

def calculate_entropy(probs):
    """计算归一化熵 (Normalized Entropy)"""
    probs = np.array(probs)
    probs = probs / probs.sum()
    # 避免 log(0)
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))
    h_max = np.log2(len(probs)) if len(probs) > 1 else 1
    return h / h_max

def design_dynamic_weight(h_rel):
    """
    修正后的动态权重逻辑：
    - 当观众投票极其分散 (h_rel > 0.9) 时，说明观众无法形成有效共识，增加评委权重 (最高 0.8)
    - 当观众投票极其极端 (h_rel < 0.4) 时，说明可能存在粉丝报团等干扰，增加评委权重以平衡 (最高 0.8)
    - 在中间地带，维持或略微降低评委权重，尊重正常的公众审美。
    """
    if h_rel > 0.8:
        # 分散区间：h_rel 从 0.8 到 1.0, alpha 从 0.5 增加到 0.8
        alpha = 0.5 + 0.3 * (h_rel - 0.8) / 0.2
    elif h_rel < 0.5:
        # 极端区间：h_rel 从 0.5 到 0.0, alpha 从 0.5 增加到 0.8
        alpha = 0.5 + 0.3 * (0.5 - h_rel) / 0.5
    else:
        # 稳定区间：权重保持在 0.5 或略低 (0.45) 以增加观赏性
        alpha = 0.45
    return alpha

def run_simulation():
    # 1. 加载数据
    df = pd.read_csv('task1_reconstructed_votes.csv')
    
    results = []
    
    # 2. 按赛季和周进行模拟
    for (season, week), group in df.groupby(['Season', 'Week']):
        if len(group) < 2: continue
        
        # 归一化评委分 (Judge Share)
        judge_scores = group['Judge_Score_Raw'].values
        judge_share = judge_scores / judge_scores.sum()
        
        # 观众分 (Audience Share)
        audience_share = group['Reconstructed_Vote_Share'].values
        
        # 计算观众投票稳定性 (归一化熵)
        h_rel = calculate_entropy(audience_share)
        
        # 获取动态评委权重
        alpha_dynamic = design_dynamic_weight(h_rel)
        
        # --- 方案 A: 传统 50/50 机制 ---
        score_5050 = 0.5 * judge_share + 0.5 * audience_share
        elim_idx_5050 = np.argmin(score_5050)
        
        # --- 方案 B: 动态平衡机制 ---
        score_dynamic = alpha_dynamic * judge_share + (1 - alpha_dynamic) * audience_share
        elim_idx_dynamic = np.argmin(score_dynamic)
        
        # 记录每位选手的得分情况 (用于后续分析)
        for i, (idx, row) in enumerate(group.iterrows()):
            results.append({
                'Season': season,
                'Week': week,
                'Contestant': row['Contestant'],
                'Judge_Score': judge_scores[i],
                'Judge_Rank': group['Judge_Score_Raw'].rank(ascending=False).iloc[i],
                'Audience_Share': audience_share[i],
                'H_Rel': h_rel,
                'Alpha_Judge': alpha_dynamic,
                'Score_5050': score_5050[i],
                'Score_Dynamic': score_dynamic[i],
                'Is_Elim_Actual': row['Is_Eliminated'],
                'Is_Elim_5050': (i == elim_idx_5050),
                'Is_Elim_Dynamic': (i == elim_idx_dynamic)
            })

    df_sim = pd.DataFrame(results)
    
    # 3. 评估指标计算
    # 指标 1: "专业遗珠" (Heartbreak Eliminations)
    # 定义: 评委排名前 25% 的选手被淘汰
    q_judge = df_sim['Judge_Rank'].quantile(0.25)
    
    def calc_metrics(df_sub, elim_col):
        total_elim = df_sub[elim_col].sum()
        # 专业遗珠: 评委分很高但被淘汰
        heartbreak = df_sub[df_sub[elim_col] & (df_sub['Judge_Rank'] <= 2)].shape[0]
        # 实力垫底但晋级: 评委分最低但没被淘汰
        bottom_save = df_sub[~df_sub[elim_col] & (df_sub['Judge_Rank'] == df_sub.groupby(['Season', 'Week'])['Judge_Rank'].transform('max'))].shape[0]
        return heartbreak, bottom_save

    h_5050, b_5050 = calc_metrics(df_sim, 'Is_Elim_5050')
    h_dyn, b_dyn = calc_metrics(df_sim, 'Is_Elim_Dynamic')
    
    print("--- Simulation Results ---")
    print(f"Total Weeks Simulated: {len(df_sim[['Season', 'Week']].drop_duplicates())}")
    print(f"\n[Baseline 50/50 System]:")
    print(f"  - Heartbreak Eliminations (Top Judges' Rank Eliminated): {h_5050}")
    print(f"  - Bottom-Judge Saves (Lowest Judge Rank Stayed): {b_5050}")
    
    print(f"\n[Proposed Dynamic Balance System]:")
    print(f"  - Heartbreak Eliminations: {h_dyn}")
    print(f"  - Bottom-Judge Saves: {b_dyn}")
    
    # 4. 可视化权重分布
    plt.figure(figsize=(10, 6))
    h_range = np.linspace(0, 1, 100)
    a_range = [design_dynamic_weight(h) for h in h_range]
    plt.plot(h_range, a_range, label='Judge Weight (Alpha)', color='blue', lw=2)
    plt.axhline(0.5, color='red', linestyle='--', label='Baseline (0.5)')
    plt.title('Dynamic Weight Logic: Judge Weight vs. Audience Voting Entropy')
    plt.xlabel('Audience Voting Entropy (Low=Extreme, High=Scattered)')
    plt.ylabel('Judge Weight Share')
    plt.legend()
    plt.savefig('task4_weight_logic.png')
    
    # 5. 案例分析：找出机制起作用的典型周
    diff = df_sim[df_sim['Is_Elim_5050'] != df_sim['Is_Elim_Dynamic']]
    if not diff.empty:
        print("\n--- Example of Mechanism Difference ---")
        example = diff.iloc[0]
        print(f"Season {example['Season']} Week {example['Week']}:")
        print(f"  Audience Entropy: {example['H_Rel']:.3f} -> Judge Weight: {example['Alpha_Judge']:.2f}")
        print(f"  50/50 Eliminated: {df_sim[(df_sim['Season']==example['Season']) & (df_sim['Week']==example['Week']) & df_sim['Is_Elim_5050']]['Contestant'].values[0]}")
        print(f"  Dynamic Eliminated: {df_sim[(df_sim['Season']==example['Season']) & (df_sim['Week']==example['Week']) & df_sim['Is_Elim_Dynamic']]['Contestant'].values[0]}")

    return df_sim

if __name__ == "__main__":
    df_sim = run_simulation()
    df_sim.to_csv('task4_simulation_results.csv', index=False)
