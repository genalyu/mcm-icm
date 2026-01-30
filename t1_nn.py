import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 模型定义: PopularityNet
# ==========================================
class PopularityNet(nn.Module):
    def __init__(self, input_dim):
        super(PopularityNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # 用于 MC Dropout 估计不确定性
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 预测一个 0-1 之间的“相对人气得分”
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 数据预处理
# ==========================================
def parse_week_judge_columns(df: pd.DataFrame):
    pat = re.compile(r"week(\d+)_judge(\d+)_score") 
    cols = [] 
    for c in df.columns: 
        m = pat.match(c) 
        if m: 
            cols.append((int(m.group(1)), int(m.group(2)), c)) 
    cols.sort(key=lambda x: (x[0], x[1])) 
    weeks = sorted({w for w, j, c in cols}) 
    return cols, weeks

def preprocess_data(df, week_judge_cols):
    # 1. 基本静态特征
    df['celebrity_industry'] = df['celebrity_industry'].fillna('Unknown')
    df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(df['celebrity_age_during_season'].mean())
    df['is_international'] = df['celebrity_homecountry/region'].apply(lambda x: 1 if pd.notna(x) and x != 'United States' else 0)
    
    # 2. 行业特征哑变量
    df['is_actor'] = df['celebrity_industry'].str.contains('Actor|Actress', na=False).astype(int)
    df['is_athlete'] = df['celebrity_industry'].str.contains('Athlete|Sport', na=False).astype(int)
    df['is_musician'] = df['celebrity_industry'].str.contains('Musician|Singer', na=False).astype(int)
    
    # 3. 舞伴实力特征 (Partner Strength)
    partner_stats = df.groupby('ballroom_partner')['placement'].mean().to_dict()
    df['partner_avg_placement'] = df['ballroom_partner'].map(partner_stats).fillna(df['placement'].mean())
    
    # 4. 初始表现特征 (Week 1)
    w1_cols = [c for (wk, j, c) in week_judge_cols if wk == 1]
    for col in w1_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['w1_total'] = df[w1_cols].sum(axis=1)
    df['w1_std'] = df[w1_cols].std(axis=1).fillna(0)
    
    # 5. 编码与归一化
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = enc.fit_transform(df[['celebrity_industry']])
    
    scaler = StandardScaler()
    num_cols = [
        'celebrity_age_during_season', 'season', 'is_international', 
        'is_actor', 'is_athlete', 'is_musician',
        'partner_avg_placement', 'w1_total', 'w1_std'
    ]
    num_features_scaled = scaler.fit_transform(df[num_cols])
    
    features_np = np.hstack([industry_encoded, num_features_scaled])
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    
    for _, _, col in week_judge_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return features_tensor, df

# ==========================================
# 3. 训练逻辑 (Learning to Rank)
# ==========================================
def train_ranking_model(df, features, week_judge_cols, epochs=200):
    input_dim = features.shape[1]
    model = PopularityNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # MarginRankingLoss: 强制要求正样本分数 > 负样本分数 + margin
    criterion = nn.MarginRankingLoss(margin=0.05)
    
    seasons = df['season'].unique()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pair_count = 0
        
        for s in seasons:
            idx_s = df[df['season'] == s].index
            df_s = df.loc[idx_s]
            feat_s = features[idx_s]
            
            # 计算每周的评委总分
            for w in range(1, 15): # 假设最大 14 周
                cols_w = [c for (wk, j, c) in week_judge_cols if wk == w]
                if not cols_w: continue
                
                # 当前周在场的选手
                scores_w = df_s[cols_w].sum(axis=1)
                active_mask = scores_w > 0
                if active_mask.sum() < 2: continue
                
                # 找出谁在下周消失了 (被淘汰)
                cols_next = [c for (wk, j, c) in week_judge_cols if wk == w + 1]
                if not cols_next: continue
                scores_next = df_s[cols_next].sum(axis=1)
                eliminated_mask = (scores_w > 0) & (scores_next == 0)
                survivor_mask = (scores_w > 0) & (scores_next > 0)
                
                if eliminated_mask.sum() == 0 or survivor_mask.sum() == 0:
                    continue
                
                # 训练对：(幸存者, 淘汰者)
                # 幸存者的 TotalScore 应该大于 淘汰者的 TotalScore
                # TotalScore = 评委分归一化 + NN人气输出
                norm_scores_w = scores_w / (scores_w.max() + 1e-9)
                
                pop_preds = model(feat_s).squeeze()
                total_scores = norm_scores_w.values + pop_preds.detach().numpy() # 简化处理，实际上应全用 tensor
                
                # 构造 Tensor 参与反向传播
                surv_indices = np.where(survivor_mask)[0]
                elim_indices = np.where(eliminated_mask)[0]
                
                # 每一对 (幸存者, 淘汰者)
                week_loss = 0
                for si in surv_indices:
                    for ei in elim_indices:
                        s_total = (torch.tensor(norm_scores_w.iloc[si], dtype=torch.float32) + pop_preds[si])
                        e_total = (torch.tensor(norm_scores_w.iloc[ei], dtype=torch.float32) + pop_preds[ei])
                        
                        # target=1 表示 s_total 应该大于 e_total
                        loss = criterion(s_total.unsqueeze(0), e_total.unsqueeze(0), torch.tensor([1.0]))
                        week_loss += loss
                        pair_count += 1
                
                if pair_count > 0 and isinstance(week_loss, torch.Tensor):
                    optimizer.zero_grad()
                    week_loss.backward()
                    optimizer.step()
                    epoch_loss += week_loss.item()
                        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/(pair_count+1e-9):.4f}")
            
    return model

# ==========================================
# 4. 评估与不确定性
# ==========================================
def evaluate_model(model, df, features, week_judge_cols):
    """计算一致性和不确定性 (MC Dropout)"""
    model.train() # 保持 Dropout 开启以进行 MC Dropout
    n_trials = 20
    
    all_pop_preds = []
    for _ in range(n_trials):
        with torch.no_grad():
            preds = model(features).squeeze().numpy()
            all_pop_preds.append(preds)
    
    all_pop_preds = np.array(all_pop_preds) # (n_trials, n_samples)
    mean_pop = all_pop_preds.mean(axis=0)
    std_pop = all_pop_preds.std(axis=0) # 不确定性度量
    
    # 计算一致性
    correct_eliminations = 0
    total_eliminations = 0
    
    # 累积平均分缓存
    cum_scores = {idx: [] for idx in df.index}
    
    seasons = df['season'].unique()
    for s in seasons:
        idx_s = df[df['season'] == s].index
        df_s = df.loc[idx_s]
        pop_s = mean_pop[idx_s]
        
        for w in range(1, 15):
            cols_w = [c for (wk, j, c) in week_judge_cols if wk == w]
            if not cols_w: continue
            
            scores_w = df_s[cols_w].sum(axis=1)
            active_mask = scores_w > 0
            if active_mask.sum() < 2: continue
            
            # 真实淘汰者
            cols_next = [c for (wk, j, c) in week_judge_cols if wk == w + 1]
            if not cols_next: continue
            scores_next = df_s[cols_next].sum(axis=1)
            real_eliminated = df_s[(scores_w > 0) & (scores_next == 0)].index
            
            if len(real_eliminated) == 0: continue
            
            # 模型预测淘汰者 (Total Score 最低的人)
            norm_scores_w = scores_w / (scores_w.max() + 1e-9)
            total_scores = norm_scores_w + pop_s
            pred_eliminated_idx = total_scores[active_mask].idxmin()
            
            if pred_eliminated_idx in real_eliminated:
                correct_eliminations += 1
            total_eliminations += 1
            
    consistency = correct_eliminations / total_eliminations if total_eliminations > 0 else 0
    uncertainty = std_pop.mean()
    
    # 收集用于相关性分析的全量数据记录
    nn_records = []
    df['predicted_pop'] = mean_pop
    
    for s in seasons:
        idx_s = df[df['season'] == s].index
        df_s = df.loc[idx_s]
        
        for w in range(1, 15):
            cols_w = [c for (wk, j, c) in week_judge_cols if wk == w]
            if not cols_w: continue
            
            scores_w = df_s[cols_w].sum(axis=1)
            active_mask = scores_w > 0
            if active_mask.sum() == 0: continue
            
            norm_scores_w = scores_w / (scores_w.max() + 1e-9)
            
            cols_next = [c for (wk, j, c) in week_judge_cols if wk == w + 1]
            is_elim_series = pd.Series(0, index=df_s.index)
            if cols_next:
                is_elim_series = (scores_w > 0) & (df_s[cols_next].sum(axis=1) == 0)
            
            # 动态特征：得分动量 (与上周比)
            prev_scores_w = pd.Series(0, index=df_s.index)
            if w > 1:
                cols_prev = [c for (wk, j, c) in week_judge_cols if wk == w - 1]
                if cols_prev:
                    prev_scores_w = df_s[cols_prev].sum(axis=1)
            
            momentum = (scores_w - prev_scores_w).fillna(0)
            avg_score_w = scores_w[active_mask].mean()
            max_score_w = scores_w[active_mask].max()
            min_score_w = scores_w[active_mask].min()
            
            rel_score = scores_w / (avg_score_w + 1e-9)
            controversy = df_s[cols_w].std(axis=1).fillna(0)
            
            # 周排名 (Judge Score Rank)
            rank_w = scores_w.rank(ascending=False, method='min')
            rel_rank_w = rank_w / active_mask.sum()
            
            # 更新累积得分
            for idx in df_s[active_mask].index:
                cum_scores[idx].append(scores_w[idx])

            for idx in df_s[active_mask].index:
                nn_records.append({
                    "Judge Score (Norm)": norm_scores_w[idx],
                    "Predicted Pop (NN)": mean_pop[idx],
                    "Total Score": norm_scores_w[idx] + mean_pop[idx],
                    "Age": df_s.loc[idx, 'celebrity_age_during_season'],
                    "Is Intl": df_s.loc[idx, 'is_international'],
                    "Partner Rank": df_s.loc[idx, 'partner_avg_placement'],
                    "W1 Score": df_s.loc[idx, 'w1_total'],
                    "Momentum": momentum[idx],
                    "Rel Score": rel_score[idx],
                    "Controversy": controversy[idx],
                    "Gap to Top": max_score_w - scores_w[idx],
                    "Gap to Bottom": scores_w[idx] - min_score_w,
                    "Week Rank": rank_w[idx],
                    "Rel Rank": rel_rank_w[idx],
                    "Cum Avg Score": np.mean(cum_scores[idx]),
                    "Cum Max Score": np.max(cum_scores[idx]),
                    "Cum Std Score": np.std(cum_scores[idx]) if len(cum_scores[idx]) > 1 else 0,
                    "Is Actor": df_s.loc[idx, 'is_actor'],
                    "Is Athlete": df_s.loc[idx, 'is_athlete'],
                    "Is Musician": df_s.loc[idx, 'is_musician'],
                    "Season": s,
                    "Is Eliminated": int(is_elim_series[idx]),
                    "Week": w
                })
    
    return consistency, uncertainty, mean_pop, std_pop, nn_records

# ==========================================
# 5. 可视化组件
# ==========================================
def plot_nn_diagnostics(df, mean_pop, std_pop, season_id=27):
    """为特定赛季绘制人气热力图和不确定性图"""
    df_s = df[df['season'] == season_id].copy()
    if df_s.empty: return
    
    # 提取该赛季的数据
    idx_s = df_s.index
    pop_s = mean_pop[idx_s]
    unc_s = std_pop[idx_s]
    names = df_s['celebrity_name'].values
    
    # 1. 绘制预测人气分布
    plt.figure(figsize=(12, 6))
    
    # 排序以便观察
    sort_idx = np.argsort(pop_s)[::-1]
    sorted_names = names[sort_idx]
    sorted_pop = pop_s[sort_idx]
    sorted_unc = unc_s[sort_idx]
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=sorted_pop, y=sorted_names, palette="viridis")
    plt.errorbar(sorted_pop, range(len(sorted_names)), xerr=sorted_unc, fmt='none', c='black', capsize=3)
    plt.title(f"Season {season_id}: Predicted Popularity (NN)")
    plt.xlabel("Popularity Score (0-1)")
    
    # 2. 绘制不确定性热力图（模拟随时间变化）
    # 注意：NN版本目前预测的是静态特征人气，这里我们展示不同行业的平均不确定性
    plt.subplot(1, 2, 2)
    industry_unc = df.groupby('celebrity_industry').apply(lambda x: std_pop[x.index].mean())
    industry_unc.sort_values().plot(kind='barh', color='salmon')
    plt.title("Mean Uncertainty by Industry")
    plt.xlabel("Avg Uncertainty (Std Dev)")
    
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_nn(nn_records):
    """为神经网络版本绘制特征相关性热力图"""
    if not nn_records: return
    corr_df = pd.DataFrame(nn_records)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("NN Comprehensive Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    try:
        df_raw = pd.read_csv(CSV_PATH)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        exit()

    week_judge_cols, weeks = parse_week_judge_columns(df_raw)
    
    features, df_processed = preprocess_data(df_raw, week_judge_cols)
    
    print("Training Ranking Model (Neural Network)...")
    model = train_ranking_model(df_processed, features, week_judge_cols, epochs=100)
    
    print("\nCalculating Metrics...")
    consistency, uncertainty, mean_pop, std_pop, nn_records = evaluate_model(model, df_processed, features, week_judge_cols)

    print("Generating Feature Correlation Heatmap...")
    plot_feature_correlation_nn(nn_records)
    
    print(f"\n{'='*30}")
    print(f"Neural Network Results:")
    print(f"Consistency (一致性): {consistency:.4f}")
    print(f"Mean Uncertainty (平均不确定性): {uncertainty:.4f}")
    print(f"{'='*30}")
    
    print("\nGenerating Plots...")
    plot_nn_diagnostics(df_processed, mean_pop, std_pop, season_id=27)
    
    print("\nNote: Uncertainty is estimated using MC Dropout variance.")
