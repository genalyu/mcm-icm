from torch import topk
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
def build_problem_percent(
        contestants, 
        J: np.ndarray, 
        ran: np.ndarray, 
        E, 
        lam: float = 2.0, 
        eps_margin: float = 1e-6 
):
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
    nvars = idx 
    bounds = [(0.0, 1.0)] * nvars 
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
    def v_of(x, t, i): 
        k = var_index.get((t, i), None) 
        return x[k] if k is not None else 0.0
    def objective(x): 
        ent_term = 0.0 
        for (t, i), k in var_index.items(): 
            v = x[k] 
            ent_term += v * math.log(max(v, 1e-12)) # v log v 
        smooth_term = 0.0 
        prev_t = None 
        for t in range(T): 
            if not ran[t]: 
                continue 
            if prev_t is None: 
                prev_t = t 
                continue 
            common = set(active_sets[prev_t].tolist()).intersection(active_sets[t].tolist()) 
            for i in common: 
                smooth_term += (v_of(x, t, i) - v_of(x, prev_t, i)) ** 2
            prev_t = t 
        return ent_term + lam * smooth_term 
    constraints = [] 
    for t in range(T): 
        if not ran[t]: 
            continue 
        inds = [var_index[(t, i)] for i in active_sets[t] if (t, i) in var_index] 
        constraints.append({ 
            "type": "eq", 
            "fun": (lambda inds=inds: (lambda x: np.sum(x[inds]) - 1.0))() 
        }) 
    for t in range(T): 
        if (not ran[t]) or (len(E[t]) == 0): 
            continue 
        act = active_sets[t].tolist() 
        eliminated = list(E[t]) 
        survivors = [j for j in act if j not in E[t]] 
        if len(survivors) == 0: 
            continue 
        for e in eliminated: 
            qe = q.get((t, e), 0.0) 
            for j in survivors: 
                qj = q.get((t, j), 0.0) 
                constraints.append({ 
                    "type": "ineq", 
                    "fun": (lambda t=t, e=e, j=j, qe=qe, qj=qj: 
                            (lambda x: (qj + v_of(x, t, j)) - (qe + v_of(x, t, e)) - eps_margin))()
                }) 
    x0 = np.zeros(nvars)
    for t in range(T): 
        if not ran[t]: 
            continue 
        act = active_sets[t] 
        if len(act) == 0: 
            continue 
        val = 1.0 / len(act) 
        for i in act: 
            x0[var_index[(t, i)]] = val 
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
    } 
    return objective, constraints, bounds, x0, meta 
def solve_percent_inverse(meta_builder_output, maxiter=2000, ftol=1e-9): 
    objective, constraints, bounds, x0, meta = meta_builder_output 
    res = minimize( 
        objective, 
        x0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=constraints, 
        options={"maxiter": maxiter, "ftol": ftol, "disp": False}, 
    ) 
    if not res.success: 
        # Return a failed result instead of raising error to allow batch processing
        return None, meta, None
    # unpack to V[t,i] 
    T, N = meta["T"], meta["N"] 
    V = np.zeros((T, N), dtype=float) 
    for (t, i), k in meta["var_index"].items(): 
        V[t, i] = res.x[k] 
    return res, meta, V

def estimate_uncertainty(contestants, J, ran, E, base_lam=2.0, n_trials=5):
    """
    通过微扰平滑参数 lam 来估计不确定性 (Uncertainty Measure)
    返回 V 的标准差矩阵
    """
    all_V = []
    # 在 base_lam 附近进行微扰
    lams = np.linspace(base_lam * 0.8, base_lam * 1.2, n_trials)
    for l in lams:
        prob = build_problem_percent(contestants, J, ran, E, lam=l)
        res, meta, V = solve_percent_inverse(prob)
        if V is not None:
            all_V.append(V)
    
    if len(all_V) < 2:
        return np.zeros_like(J)
    
    # 计算每个位置的标准差作为不确定度
    V_stack = np.stack(all_V)
    uncertainty = np.std(V_stack, axis=0)
    return uncertainty
def check_consistency_percent(meta, V):
    T, N = meta["T"], meta["N"] 
    ran = meta["ran"] 
    E = meta["E"] 
    q = meta["q"] 
    active_sets = meta["active_sets"]
    preds = {} 
    elim_weeks = [] 
    for t in range(T): 
        if not ran[t]: 
            continue 
        if len(E[t]) > 0:
            elim_weeks.append(t) 
        act = active_sets[t] 
        C = {i: (q.get((t, i), 0.0) + V[t, i]) for i in act} 
        pred = min(C, key=C.get) if len(C) else None 
        preds[t] = pred 
    correct = 0 
    for t in elim_weeks:
        if preds.get(t, None) in E[t]: 
            correct += 1 
    rate = correct / len(elim_weeks) if elim_weeks else np.nan 
    return rate, preds, elim_weeks 
def plot_heatmap_votes(meta, V, uncertainty, df_season, season_id):
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
    fig.colorbar(im0, ax=ax[0], label="Estimated vote share")
    ax[0].set_title(f"Season {season_id}: Estimated Vote Share")
    
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
    plt.title(f"Season {season_id}: Vote share trajectories for top {top_k} finalists") 
    plt.legend() 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout()
    plt.show()

def run_one_season_percent(df, season_id: int, week_judge_cols, max_week: int, lam: float = 2.0, silent=False):
    df_s = df[df["season"] == season_id].copy() 
    if df_s.empty: 
        return None
    
    contestants, J, ran = season_matrices(df_s, week_judge_cols, max_week) 
    E = elimination_set(J, ran) 
    problem = build_problem_percent(contestants, J, ran, E, lam=lam) 
    res, meta, V = solve_percent_inverse(problem) 
    
    if V is None:
        return None
        
    uncertainty = estimate_uncertainty(contestants, J, ran, E, base_lam=lam)
    rate, preds, elim_weeks = check_consistency_percent(meta, V) 
    
    if not silent:
        print(f"[Season {season_id}] Success. Consistency = {rate:.3f}, Avg Uncertainty = {np.nanmean(uncertainty):.4f}") 
        plot_heatmap_votes(meta, V, uncertainty, df_s, season_id) 
        plot_finalists_trajectories(meta, V, df_s, season_id, top_k=3) 
        
    return {"season": season_id, "consistency": rate, "uncertainty": np.nanmean(uncertainty)}

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    
    all_seasons = sorted(df["season"].unique())
    results = []
    
    print(f"{'Season':<10} | {'Consistency':<12} | {'Uncertainty':<12}")
    print("-" * 40)
    
    for sid in all_seasons:
        # Run silent for all, only plot one example (e.g., season 27)
        res = run_one_season_percent(df, sid, week_judge_cols, max_week, silent=(sid != 27))
        if res:
            results.append(res)
            print(f"{res['season']:<10} | {res['consistency']:<12.3f} | {res['uncertainty']:<12.4f}")

    rdf = pd.DataFrame(results)
    print("\nSummary Statistics:")
    print(f"Mean Consistency: {rdf['consistency'].mean():.3f}")
    print(f"Mean Uncertainty: {rdf['uncertainty'].mean():.4f}")