import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from functools import partial

def solve_fan_votes_strictly(
    judge_scores_raw,  # list or np.array: 本周所有人的评委原始分
    eliminated_idx,    # int: 被淘汰者的索引 (0-based)
    prior_votes,       # list or np.array: XGBoost 的预测值 (作为初值)
    rule='percent',    # 'percent' or 'rank'
    epsilon=1e-5       # 安全边际，确保不等式严格成立
):
    """
    使用 SLSQP 算法求解满足淘汰约束的观众票数分布。
    目标：在满足淘汰结果的前提下，让 V 尽可能接近 XGBoost 的预测。
    """
    n = len(judge_scores_raw)
    
    # ==========================================
    # 1. 数据预处理 (把 J 统一成 "分数" 概念)
    # ==========================================
    if rule == 'rank':
        # 如果是 Rank 制，我们把评委分先转成 Rank (1..N)
        # 然后把 Rank 当作数值来处理
        # method='min' 表示同分时取最小排名，这更符合“同分劣势”
        J_processed = rankdata(judge_scores_raw, method='min')
    else:
        # 如果是 Percent 制，直接用原始分
        J_processed = np.array(judge_scores_raw, dtype=float)
        
    # 归一化 J (关键步骤！让 J 和 V 在同一个数量级)
    # 这样 J + V 的时候权重才大概对等
    j_sum = np.sum(J_processed)
    if j_sum == 0: j_sum = 1
    J_norm = J_processed / j_sum

    # ==========================================
    # 2. 定义目标函数 (Objective)
    # ==========================================
    # 我们希望 V_final 和 prior_votes 越像越好
    # 如果没有 prior，就假设是均匀分布
    if prior_votes is None:
        prior_votes = np.ones(n) / n
    else:
        prior_votes = np.array(prior_votes)
        if np.sum(prior_votes) == 0: 
             prior_votes = np.ones(n) / n
        else:
             prior_votes = prior_votes / np.sum(prior_votes) # 确保初值归一化

    def objective(v):
        return np.sum((v - prior_votes)**2)

    # ==========================================
    # 3. 定义约束条件 (Constraints)
    # ==========================================
    constraints = []

    # C1: 等式约束 -> 所有 V 加起来必须等于 1
    constraints.append({'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0})

    # C2: 不等式约束 -> 淘汰逻辑
    # 逻辑：对于每一个幸存者 S，他的总分必须 > 被淘汰者 E 的总分
    # Score = 0.5 * J + 0.5 * V (我们假设权重相等，这在 Task 1 足够了)
    # 即: (J_s + V_s) - (J_e + V_e) > epsilon
    
    # 这是一个帮助函数，用于生成约束
    def constraint_func(v, s_idx, e_idx, J_vals, eps):
        score_survivor = J_vals[s_idx] + v[s_idx]
        score_eliminated = J_vals[e_idx] + v[e_idx]
        # 优化器要求 fun(x) >= 0，所以我们要写成:
        # (S的分 - E的分 - epsilon) >= 0
        return score_survivor - score_eliminated - eps

    # 遍历所有幸存者
    for i in range(n):
        if i == eliminated_idx:
            continue
        
        # 【极其重要】使用 partial 锁死变量！
        # 如果你直接写 lambda v: ... i ...，Python 会只记住最后一次循环的 i
        # 这是造成 17% 准确率的元凶！
        cons = {
            'type': 'ineq', 
            'fun': partial(constraint_func, s_idx=i, e_idx=eliminated_idx, J_vals=J_norm, eps=epsilon)
        }
        constraints.append(cons)

    # ==========================================
    # 4. 执行优化 (SLSQP)
    # ==========================================
    # 边界：0 <= V <= 1
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    # 初始猜测：直接用 prior
    x0 = prior_votes

    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
    )

    # ==========================================
    # 5. 结果处理与兜底
    # ==========================================
    if result.success:
        return result.x / np.sum(result.x) # 再次归一化保平安
    else:
        # 如果优化失败（极其罕见，除非评委分极其离谱导致无解）
        # 我们启动“暴力修正模式” (Heuristic Fallback)
        # 这就是我上次给你的那个简单逻辑，作为备用轮胎
        print(f"Warning: Optimization failed for ElimIdx {eliminated_idx}. Using fallback.")
        v_fallback = prior_votes.copy()
        # 强行把淘汰者的票压低
        v_fallback[eliminated_idx] *= 0.1 
        # 把票分给其他人
        diff = 1.0 - np.sum(v_fallback)
        v_fallback += diff / n
        return v_fallback / np.sum(v_fallback)

# 测试代码
if __name__ == "__main__":
    # 模拟一个场景：S1, S2, S3. 
    # S3 (索引2) 是淘汰者。
    # S3 的评委分很高 (30)，S1 评委分很低 (20)。这需要 S3 的观众票极低才能死。
    J = [20, 25, 30] 
    prior = [0.3, 0.3, 0.4] # XGB 猜错了，觉得 S3 人气高
    elim = 2 # S3 被淘汰
    
    V_final = solve_fan_votes_strictly(J, elim, prior, rule='percent')
    
    print("Judge Scores:", J)
    print("Prior Votes:", prior)
    print("Final Votes:", np.round(V_final, 4))
    
    # 验证
    J_norm = np.array(J) / np.sum(J)
    Scores = J_norm + V_final
    print("Final Scores:", Scores)
    print("Is Eliminated Score Lowest?", Scores[elim] < np.min(Scores[:2]))