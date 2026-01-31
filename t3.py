import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from t1_gas import load_data, parse_week_judge_columns, run_one_season_percent

def analyze_problem_3(all_records):
    print("\n--- Problem 3: Feature Impact Analysis ---")
    if not all_records:
        print("No records found for analysis.")
        return
    
    rdf = pd.DataFrame(all_records)
    
    # 定义要分析的特征
    features = ['Age', 'Is Intl', 'Partner Strength', 'Is Actor', 'Is Athlete', 'Is Musician']
    
    # 预处理：归一化特征
    for f in features:
        if rdf[f].std() > 0:
            rdf[f] = (rdf[f] - rdf[f].mean()) / rdf[f].std()
        else:
            rdf[f] = 0.0
            
    # 目标 1: 评委评分的影响因素
    X = sm.add_constant(rdf[features])
    model_judge = sm.OLS(rdf['Judge Score (Norm)'], X).fit()
    
    # 目标 2: 观众投票的影响因素
    model_audience = sm.OLS(rdf['Est. Vote Share'], X).fit()
    
    # 结果对比
    comparison = pd.DataFrame({
        'Feature': features,
        'Judge Impact': model_judge.params[features],
        'Audience Impact': model_audience.params[features],
        'Judge P-value': model_judge.pvalues[features],
        'Audience P-value': model_audience.pvalues[features]
    })
    
    print("\nImpact Comparison (Regression Coefficients):")
    print(comparison)
    
    # 可视化影响差异
    plt.figure(figsize=(10, 8))
    comparison.plot(x='Feature', y=['Judge Impact', 'Audience Impact'], kind='barh', ax=plt.gca())
    plt.title("Impact of Features: Judge Scores vs Audience Votes")
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig("problem_3_impact.png")
    print("Saved problem_3_impact.png")
    
    # 统计运动员 vs 其他职业的表现
    athlete_stats = rdf.groupby('Is Athlete')[['Judge Score (Norm)', 'Est. Vote Share']].mean()
    print("\nAthlete (1) vs Others (0) Performance Mean:")
    print(athlete_stats)

if __name__ == "__main__":
    CSV_PATH = "2026_MCM_Problem_C_Data.csv"
    df = load_data(CSV_PATH)
    week_judge_cols, weeks = parse_week_judge_columns(df)
    max_week = max(weeks)
    
    all_records = []
    print("Collecting data from all seasons for analysis...")
    # all_seasons = sorted(df["season"].unique())
    all_seasons = [1]
    
    for sid in all_seasons:
        res = run_one_season_percent(df, sid, week_judge_cols, max_week, silent=True)
        if res:
            all_records.extend(res["records"])
            
    analyze_problem_3(all_records)
