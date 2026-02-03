# 🕺 Dancing with the Stars (DWTS) Voting System Optimization

This project aims to analyze and optimize the scoring system of the "Dancing with the Stars" (DWTS) television show. By leveraging machine learning and statistical modeling, we reconstruct latent voting patterns, simulate counterfactual scenarios, and propose a new, fairer scoring mechanism called the **Causal-Adaptive Truncation Protocol (CATP)**.

---

## 🏗️ Project Structure & Methodology

Our work is divided into four main tasks, following a rigorous data-driven pipeline:

### **🛠️ Data Processing (Foundation)**
- **Data Cleaning:** Preprocessing raw competition data and historical voting records.
- **Topological Mapping:** Mapping contestants, partners, and seasons into a consistent analytical framework.
- **Feature Engineering:** Extracting key features such as celebrity industry, age, and professional dancer historical performance.

### **Task 1: 🗳️ Latent Vote Reconstruction**
- **Goal:** Estimate the hidden audience vote shares that are not explicitly provided in the raw data.
- **Model:** 🤖
    - **Prior Estimation:** Utilizing **XGBoost** to predict initial vote distributions based on historical features.
    - **Posterior Refinement:** Applying **SLSQP (Sequential Least Squares Programming)** optimization to align predictions with known elimination outcomes.
- **Output:** Reconstructed Fan Votes ($\hat{V}$).

### **Task 2: 🎭 Counterfactual Simulation**
- **Goal:** Analyze "What-If" scenarios to identify historical anomalies where popularity overshadowed performance.
- **Dual Pathway Analysis:**
    - **Rank-based (RM):** Evaluating shifts in standings under different weight distributions.
    - **Percentage-based (PM):** Simulating fate divergence timelines.
- **Key Output:** Identification of the **"Jerry Rice Zone"**—a state where high popularity creates an insurmountable barrier for higher-skilled contestants.

### **Task 3: 🔍 Identifying Impact Factors**
- **Goal:** Quantify the drivers of voting bias and judge preferences.
- **Analytical Tools:**
    - **OLS Baseline:** Linear regression to identify primary correlations.
    - **Random Forest & SHAP:** Non-linear modeling to interpret complex feature interactions (e.g., the "Kingmaker" effect).
    - **Controversy Analysis:** Measuring the gap between judge scores and audience votes.
- **Finding:** The **"Kingmaker" Effect**, where professional partners significantly influence the audience's voting behavior regardless of the celebrity's skill.

### **Task 4: 🚀 Scoring System Optimization**
- **Goal:** Design a mechanism to mitigate extreme popularity bias while maintaining audience engagement.
- **Proposed Mechanism: CATP (Causal-Adaptive Truncation Protocol)**
- **Formula:** 🔢

$$
V_i' = \begin{cases} 
V_i^{actual} & \text{if } R_i \le \sigma_i \\ 
\hat{V}_i + \sigma_i + \gamma \cdot \ln(1 + R_i - \sigma_i) & \text{if } R_i > \sigma_i 
\end{cases}
$$

Where $R_i$ is the residual (popularity bias), and $\sigma_i$ is the dynamic threshold.
- **Evaluation:** Sensitivity analysis demonstrates high feasibility and robustness in correcting anomalies like Jerry Rice and Bobby Bones without "punishing" fair popularity.

---

## 📂 Repository Overview

### 💻 Core Scripts
- `run_reconstruction.py`: Main script for Task 1 vote reconstruction.
- `t2_enhanced.py`: Enhanced counterfactual simulation for Task 2.
- `task3+4.ipynb`: Interactive analysis for impact factors and the CATP mechanism evaluation.
- `task_global_sensitivity.py`: Global sensitivity analysis for model parameters.

### 📊 Data & Results
- `2026_MCM_Problem_C_Data.csv`: Raw input data.
- `task1_reconstructed_votes.csv`: Reconstructed voting results.
- `images/`: Visualization outputs including feature correlations, vote trajectories, and sensitivity heatmaps.

---

## 📝 Summary
Our solution provides a comprehensive framework to balance professional judgment with audience participation, ensuring that "Dancing with the Stars" remains both a legitimate competition and a popular entertainment spectacle.
