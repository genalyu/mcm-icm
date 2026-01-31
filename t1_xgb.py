import re
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

# --- 引入 t1_gas.py 的核心逻辑用于生成银标 (Silver Label) ---

def get_silver_labels_v(meta):
    """
    使用逆向优化逻辑，为训练集生成符合 50/50 规则的 V (观众投票份额)。
    V 满足 Sum(V) = 1, V >= 0, 且 0.5*q + 0.5*V 使得淘汰者分数最低。
    """
    T, N = meta["T"], meta["N"]
    ran, E, q = meta["ran"], meta["E"], meta["q"]
    active_sets = meta["active_sets"]
    
    v_map = {}
    
    for t in range(T):
        if not ran[t] or len(active_sets[t]) == 0:
            continue
            
        act = active_sets[t]
        n_act = len(act)
        
        # 提取当前周的 q (已经归一化，Sum(q)=1)
        q_t = np.array([q.get((t, i), 1.0/n_act) for i in act])
        
        # 被淘汰者的索引（在 act 中的位置）
        eliminated_list = list(E[t])
        if not eliminated_list:
            # 如果没淘汰，V 取均匀分布作为银标
            for idx, i in enumerate(act):
                v_map[(t, i)] = 1.0 / n_act
            continue
            
        e_val = eliminated_list[0]
        if e_val not in act:
            # 异常情况
            for idx, i in enumerate(act):
                v_map[(t, i)] = 1.0 / n_act
            continue
            
        e_idx = list(act).index(e_val)
        
        # 优化目标：使 V 尽可能接近均匀分布，同时满足淘汰约束
        # 变量 v 是当前周所有活跃选手的投票份额
        def objective(v):
            return 0.5 * np.sum((v - 1.0/n_act)**2)

        def constraint_sum(v):
            return np.sum(v) - 1.0

        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        # 淘汰约束: 0.5*q[e] + 0.5*V[e] <= 0.5*q[o] + 0.5*V[o] - eps
        # 简化为: V[e] - V[o] <= q[o] - q[e] - 2*eps
        for o_idx in range(n_act):
            if o_idx == e_idx: continue
            
            def constr_elim(v, o=o_idx, e=e_idx):
                # q_t[o] + v[o] >= q_t[e] + v[e] + margin
                return (q_t[o] + v[o]) - (q_t[e] + v[e]) - 0.001
            
            constraints.append({'type': 'ineq', 'fun': constr_elim})

        # 边界: 0 <= v <= 1
        bounds = [(0, 1) for _ in range(n_act)]
        v0 = np.ones(n_act) / n_act
        
        res = minimize(objective, v0, method='SLSQP', constraints=constraints, bounds=bounds, options={'maxiter': 50})
        
        if res.success:
            final_v = res.x
        else:
            # 如果优化失败，尝试一个极端的 V (淘汰者为0，其他人平分)
            final_v = np.ones(n_act) / (n_act - 1)
            final_v[e_idx] = 0.0
            
        for idx, i in enumerate(act):
            v_map[(t, i)] = final_v[idx]
            
    return v_map

def load_data(csv_path: str) -> pd.DataFrame: 
    df = pd.read_csv(csv_path)
    return df

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

def season_matrices(df_season: pd.DataFrame, week_judge_cols, max_week: int):
    contestants = df_season["celebrity_name"].tolist() 
    N = len(contestants) 
    T = max_week 
    J = np.full((T, N), np.nan, dtype=float) 
    ran = np.zeros(T, dtype=bool) 
    for t in range(1, T + 1): 
        cols_t = [c for (w, j, c) in week_judge_cols if w == t] 
        mat = df_season[cols_t].to_numpy(dtype=float) 
        if np.all(np.isnan(mat)): 
            ran[t - 1] = False 
            continue 
        ran[t - 1] = True 
        J[t - 1, :] = np.nansum(mat, axis=1) 
    return contestants, J, ran 

def active_mask(J_t: np.ndarray):
    return np.isfinite(J_t) & (J_t > 0) 

def elimination_set(J: np.ndarray, ran: np.ndarray):
    T, N = J.shape 
    E = [set() for _ in range(T)] 
    for t in range(T - 1): 
        if (not ran[t]) or (not ran[t + 1]): 
            continue 
        act_t = active_mask(J[t]) 
        act_t1 = active_mask(J[t + 1]) 
        eliminated = np.where(act_t & (~act_t1))[0] 
        E[t] = set(eliminated.tolist()) 
    return E   

def extract_features(meta):
    """
    从 meta 信息中提取特征用于 XGBoost 模型。
    """
    T, N = meta["T"], meta["N"]
    J = meta["J"]
    active_sets = meta["active_sets"]
    contestants = meta["contestants"]
    df_s = meta["df_season"]
    partner_stats = meta.get("partner_stats", {})
    
    features = []
    indices = []
    
    # 预计算一些选手级特征
    celebrity_info = {}
    for i, name in enumerate(contestants):
        row = df_s[df_s['celebrity_name'] == name].iloc[0]
        celebrity_info[i] = {
            "age": row.get('celebrity_age_during_season', 40),
            "is_intl": 1 if (pd.notna(row.get('celebrity_homecountry/region')) and row.get('celebrity_homecountry/region') != 'United States') else 0,
            "is_actor": 1 if 'Actor' in str(row.get('celebrity_industry', '')) else 0,
            "is_athlete": 1 if 'Athlete' in str(row.get('celebrity_industry', '')) else 0,
            "partner_rank": partner_stats.get(row.get('ballroom_partner', ''), 5.0)
        }

    # 预计算选手赛季内的动态特征
    resilience_stats = {i: {"low_score_survivals": 0, "rel_scores": []} for i in range(N)}
    
    for t in range(T):
        if not meta["ran"][t]:
            continue
        act = active_sets[t]
        if len(act) == 0:
            continue
            
        scores_t = J[t, act]
        avg_score = np.mean(scores_t)
        max_score = np.max(scores_t)
        min_score = np.min(scores_t)
        std_score = np.std(scores_t) + 1e-9
        
        # 判定“低分”阈值（比如后 30%）
        low_threshold = np.percentile(scores_t, 30) if len(scores_t) > 3 else min_score
        
        sorted_indices = np.argsort(-scores_t)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(scores_t)) + 1
        rank_map = {act[idx]: ranks[idx] for idx in range(len(act))}

        for i in act:
            score = J[t, i]
            rank = rank_map[i]
            info = celebrity_info[i]
            res = resilience_stats[i]
            
            # 1. 基础评分特征
            feat = [
                score,                      
                score / (avg_score + 1e-9), 
                max_score - score,          
                score - min_score,          
                (score - avg_score) / std_score, 
                rank,                       
                rank / len(act),            
                len(act)                    
            ]
            
            # 2. 趋势与历史表现 (Judge)
            for lookback in [1, 2]:
                prev_t = t - lookback
                if prev_t >= 0 and (prev_t, i) in meta["var_index"]:
                    feat.append(J[prev_t, i] - score) 
                else:
                    feat.append(0.0)
            
            all_prev = [J[pt, i] for pt in range(t+1) if (pt, i) in meta["var_index"]]
            feat.append(np.mean(all_prev))
            feat.append(np.min(all_prev))
            
            # 3. 动态生存韧性特征 (新加入)
            feat.extend([
                res["low_score_survivals"],                # 之前低分幸存的次数
                np.mean(res["rel_scores"]) if res["rel_scores"] else 1.0, # 赛季至今平均相对表现
                res["rel_scores"][-1] if res["rel_scores"] else 1.0       # 上周相对表现
            ])
            
            # 更新选手的本周数据，供下周使用
            res["rel_scores"].append(score / (avg_score + 1e-9))
            if score <= low_threshold and i not in meta["E"][t]:
                res["low_score_survivals"] += 1

            # 4. 静态背景特征
            feat.extend([
                info["age"],
                info["is_intl"],
                info["is_actor"],
                info["is_athlete"],
                info["partner_rank"]
            ])
                
            features.append(feat)
            indices.append((t, i))
            
    return np.array(features), indices

def build_problem_percent(
        contestants, 
        J: np.ndarray, 
        ran: np.ndarray, 
        E, 
        df_season: pd.DataFrame = None,
        partner_stats: dict = None,
        lam: float = 2.0, 
        eps_margin: float = 1e-6 
):
    """
    为了保持接口一致，返回与 t1_gas 类似的结构。
    但在 XGBoost 版本中，主要工作是准备特征和 meta。
    """
    T, N = J.shape 
    var_index = {} 
    idx = 0 
    active_sets = {}
    for t in range(T): 
        if not ran[t]: 
            continue 
        act = active_mask(J[t]) 
        active_sets[t] = np.where(act)[0] 
        for i in active_sets[t]: 
            var_index[(t, i)] = idx 
            idx += 1 
            
    q = {} 
    for t in range(T): 
        if not ran[t]: 
            continue 
        act = active_sets[t] 
        denom = np.nansum(J[t, act]) 
        if not np.isfinite(denom) or denom <= 0: 
            continue 
        for i in act: 
            q[(t, i)] = J[t, i] / denom 

    meta = { 
        "T": T, 
        "N": N, 
        "contestants": contestants, 
        "J": J, 
        "ran": ran,
        "E": E, 
        "q": q, 
        "active_sets": active_sets, 
        "var_index": var_index, 
        "df_season": df_season,
        "partner_stats": partner_stats
    }
    
    # 构造 XGBoost 的训练目标 y
    X, feat_indices = extract_features(meta)
    y = []
    for t, i in feat_indices:
        # 如果在该周被淘汰，目标值设为 0，否则设为 1
        if i in E[t]:
            y.append(0)
        else:
            y.append(1)
    
    return X, np.array(y), feat_indices, meta

def solve_percent_inverse(meta_builder_output, random_state=42, X_train=None, y_train=None, pretrained_model=None): 
    """
    使用 XGBoost 原生接口预测每个选手的“受欢迎程度”，并将其转化为投票份额 V。
    支持传入预训练模型。
    """
    X_orig, y_orig, feat_indices, meta = meta_builder_output
    
    if pretrained_model is not None:
        model = pretrained_model
    else:
        # 如果没有传入训练集，使用单赛季数据（不推荐，样本太少）
        X = X_train if X_train is not None else X_orig
        y = y_train if y_train is not None else y_orig
        dtrain = xgb.DMatrix(X, label=y)
        
        # 为单赛季准备 group
        weeks = sorted(list(set([t for t, i in feat_indices])))
        groups = []
        for t in weeks:
            groups.append(len([1 for pt, pi in feat_indices if pt == t]))
        dtrain.set_group(groups)

        params = {
            'objective': 'rank:pairwise',
            'max_depth': 3,
            'eta': 0.1,
            'seed': random_state,
            'verbosity': 0
        }
        model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 在原始特征上进行预测
    dorig = xgb.DMatrix(X_orig)
    # 排名模型预测出的是 rank score
    preds = model.predict(dorig)
    
    # 将预测值映射回 V[t, i] 空间
    T, N = meta["T"], meta["N"]
    V = np.zeros((T, N), dtype=float)
    
    for t in range(T):
        if not meta["ran"][t]:
            continue
        
        act = meta["active_sets"][t]
        week_scores = []
        week_indices = []
        
        for idx, (pt, pi) in enumerate(feat_indices):
            if pt == t:
                week_scores.append(preds[idx])
                week_indices.append(pi)
        
        if week_scores:
            week_scores = np.array(week_scores)
            # 使用 Softmax 将 rank scores 转化为概率分布 (投票份额)
            # 减去 max 提高数值稳定性
            exp_s = np.exp(week_scores - np.max(week_scores))
            normalized_v = exp_s / np.sum(exp_s)
            
            for i, v_val in zip(week_indices, normalized_v):
                V[t, i] = v_val
                
    class MockRes:
        def __init__(self, success):
            self.success = success
    
    return MockRes(True), meta, V

def estimate_uncertainty(contestants, J, ran, E, df_season=None, partner_stats=None, n_trials=5):
    """
    通过 Bootstrap 训练多个模型来估计预测结果的标准差
    """
    all_V = []
    # 准备原始数据
    X, y, feat_indices, meta = build_problem_percent(contestants, J, ran, E, df_season=df_season, partner_stats=partner_stats)
    
    for i in range(n_trials):
        # 自助抽样
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # 使用不同的训练集和随机种子进行预测
        _, _, V_boot = solve_percent_inverse((X, y, feat_indices, meta), 
                                            random_state=i, 
                                            X_train=X_boot, 
                                            y_train=y_boot)
        all_V.append(V_boot)
        
    V_stack = np.stack(all_V)
    return np.std(V_stack, axis=0)

def check_consistency_percent(meta, V):
    T, N = meta["T"], meta["N"] 
    ran = meta["ran"] 
    E = meta["E"] 
    q = meta["q"] 
    active_sets = meta["active_sets"]
    preds_model = {} 
    preds_judge = {} 
    elim_weeks = [] 
    for t in range(T): 
        if not ran[t]: 
            continue 
        if len(E[t]) > 0:
            elim_weeks.append(t) 
        act = active_sets[t] 
        # 模型预测：q + V
        C_model = {i: (q.get((t, i), 0.0) + V[t, i]) for i in act} 
        pred_m = min(C_model, key=C_model.get) if len(C_model) else None 
        preds_model[t] = pred_m
        
        # 基准预测：仅靠评委分 q
        C_judge = {i: q.get((t, i), 0.0) for i in act}
        pred_j = min(C_judge, key=C_judge.get) if len(C_judge) else None
        preds_judge[t] = pred_j

    correct_m = 0 
    correct_j = 0
    for t in elim_weeks:
        if preds_model.get(t, None) in E[t]: 
            correct_m += 1 
        if preds_judge.get(t, None) in E[t]:
            correct_j += 1
            
    rate_m = correct_m / len(elim_weeks) if elim_weeks else np.nan 
    rate_j = correct_j / len(elim_weeks) if elim_weeks else np.nan
    return rate_m, rate_j, preds_model, elim_weeks 

def plot_heatmap_votes(meta, V, uncertainty, df_season, season_id):
    import matplotlib.pyplot as plt
    contestants = meta["contestants"] 
    ran = meta["ran"]
    info = df_season[["celebrity_name", "placement"]].drop_duplicates().set_index("celebrity_name")
    placement = info["placement"].to_dict() 
    order_names = sorted(contestants, key=lambda n: (placement.get(n, 999), n)) 
    order_idx = [contestants.index(n) for n in order_names] 
    ran_weeks = [t for t in range(meta["T"]) if ran[t]] 
    
    V_plot = V[ran_weeks, :][:, order_idx]
    U_plot = uncertainty[ran_weeks, :][:, order_idx]
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    im0 = ax[0].imshow(V_plot, aspect="auto", cmap="viridis")
    fig.colorbar(im0, ax=ax[0], label="Estimated vote share (XGBoost)")
    ax[0].set_title(f"Season {season_id}: Estimated Vote Share (XGB)")
    
    im1 = ax[1].imshow(U_plot, aspect="auto", cmap="Reds")
    fig.colorbar(im1, ax=ax[1], label="Uncertainty (Std Dev)")
    ax[1].set_title(f"Season {season_id}: Uncertainty Measure")
    
    for a in ax:
        a.set_yticks(range(len(ran_weeks)))
        a.set_yticklabels([f"Week {t+1}" for t in ran_weeks])
        a.set_xticks(range(len(order_names)))
        a.set_xticklabels(order_names, rotation=90)
    
    plt.tight_layout()
    plt.show()

def plot_finalists_trajectories(meta, V, df_season, season_id, top_k=3):
    import matplotlib.pyplot as plt
    contestants = meta["contestants"] 
    ran = meta["ran"] 
    T = meta["T"] 
    info = df_season[["celebrity_name", "placement"]].drop_duplicates().set_index("celebrity_name")
    placement = info["placement"].to_dict() 
    finalists = [n for n in contestants if placement.get(n, 999) <= top_k] 
    finalists = sorted(finalists, key=lambda n: placement[n]) 
    plt.figure(figsize=(10, 5)) 
    for name in finalists: 
        i = contestants.index(name) 
        series = [V[t, i] if ran[t] else np.nan for t in range(T)] 
        plt.plot(range(1, T + 1), series, marker="o", linewidth=2, label=f"{name} (#{placement[name]})")
    plt.xlabel("Week") 
    plt.ylabel("Estimated vote share") 
    plt.title(f"Season {season_id}: Vote share trajectories (XGBoost)") 
    plt.legend() 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout()
    plt.show()

def train_global_model(df, season_ids, week_judge_cols, max_week, partner_stats=None):
    """
    训练回归模型，预测由 t1_gas 逻辑生成的银标 V。
    """
    all_X = []
    all_y = []
    
    for sid in season_ids:
        df_s = df[df["season"] == sid].copy()
        if df_s.empty: continue
        
        contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week)
        E = elimination_set(J, ran)
        X, _, feat_indices, meta = build_problem_percent(contestants, J, ran, E, df_season=df_s, partner_stats=partner_stats)
        
        # 获取银标 V
        v_map = get_silver_labels_v(meta)
        
        # 将银标映射到特征行
        y_silver = []
        for (t, i) in feat_indices:
            y_silver.append(v_map.get((t, i), 0.0))
            
        all_X.append(X)
        all_y.extend(y_silver)
        
    X_train = np.vstack(all_X)
    y_train = np.array(all_y)
    
    # 转换为 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,              # 稍微增加深度
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 0.1,
        'lambda': 1.0,
        'seed': 42,
        'verbosity': 0
    }
    
    model = xgb.train(params, dtrain, num_boost_round=200)
    return model

def run_one_season_percent(df, season_id: int, week_judge_cols, max_week: int, silent=False, global_model=None, partner_stats=None):
    df_s = df[df["season"] == season_id].copy() 
    if df_s.empty: 
        return None
    
    contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week) 
    E = elimination_set(J, ran) 
    # 模型预测 V
    X_test, _, feat_indices, meta = build_problem_percent(contestants, J, ran, E, df_season=df_s, partner_stats=partner_stats)
    dtest = xgb.DMatrix(X_test)
    v_preds = global_model.predict(dtest)
    
    # 将预测的 v 映射回 (t, i) 并进行周归一化
    V = np.zeros((meta["T"], meta["N"]))
    for t in range(meta["T"]):
        if not meta["ran"][t]: continue
        
        week_indices = [idx for idx, (pt, pi) in enumerate(feat_indices) if pt == t]
        if not week_indices: continue
        
        # 提取该周的原始预测
        week_v = v_preds[week_indices]
        # Softmax 归一化，确保该周 V 之和为 1 且非负
        # 这样 V 的尺度就与归一化后的 q (Sum=1) 完全匹配，符合 50/50 规则
        exp_v = np.exp(week_v - np.max(week_v))
        norm_v = exp_v / np.sum(exp_v)
        
        for idx_in_week, global_idx in enumerate(week_indices):
            _, pi = feat_indices[global_idx]
            V[t, pi] = norm_v[idx_in_week]
    
    # 诊断：打印归一化后的 V 的范围和 q 的范围
    q_vals = list(meta["q"].values())
    if not silent:
        v_vals = V[V > 0]
        print(f"DEBUG [Season {season_id}]: q range: [{min(q_vals):.4f}, {max(q_vals):.4f}], V_norm range: [{min(v_vals):.4f}, {max(v_vals):.4f}]")
        print(f"DEBUG [Season {season_id}]: V_norm mean: {np.mean(v_vals):.4f}, std: {np.std(v_vals):.4f}")
    
    # 为不确定性估计传入必要参数
    uncertainty = estimate_uncertainty(contestants, J, ran, E, df_season=df_s, partner_stats=partner_stats)
    rate_m, rate_j, preds, elim_weeks = check_consistency_percent(meta, V) 
    
    if not silent:
        print(f"[Season {season_id} XGB] Success. Model Consist. = {rate_m:.3f}, Judge-only Consist. = {rate_j:.3f}, Avg Uncertainty = {np.nanmean(uncertainty):.4f}") 
        # plot_heatmap_votes(meta, V, uncertainty, df_s, season_id) 
        # plot_finalists_trajectories(meta, V, df_s, season_id, top_k=3) 
        
    return {
        "season": season_id, 
        "consistency": rate_m, 
        "consistency_judge": rate_j,
        "uncertainty": np.nanmean(uncertainty)
    }

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    
    # 预计算舞伴统计数据
    partner_stats = df.groupby('ballroom_partner')['placement'].mean().to_dict()
    
    # 获取所有可用的赛季
    all_season_ids = sorted(df["season"].unique())
    print(f"Available seasons: {all_season_ids}")
    
    print("\n" + "="*50)
    print("VALIDATION: Leave-One-Season-Out (L.O.S.O)")
    print("="*50)
    
    # 演示：对前 3 个赛季进行验证
    target_seasons = [1, 2, 3]
    for sid in target_seasons:
        train_ids = [i for i in all_season_ids if i != sid]
        model = train_global_model(df, train_ids, week_judge_cols, max_week, partner_stats=partner_stats)
        
        print(f"\n--- Testing Season {sid} (Model trained on others) ---")
        res = run_one_season_percent(df, sid, week_judge_cols, max_week, silent=False, global_model=model, partner_stats=partner_stats)
