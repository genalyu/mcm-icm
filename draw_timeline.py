import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据读取与预处理 (保持不变)
# ==========================================
try:
    # 尝试读取你之前生成的 CSV
    df = pd.read_csv("task2_micro_analysis.csv")
except FileNotFoundError:
    # 如果没有，用这个示例数据兜底
    print("提示：未找到 task2_micro_analysis.csv，使用示例数据进行演示。")
    data = {
        "Name": ["Jerry Rice", "Billy Ray Cyrus", "Bobby Bones", "Bristol Palin"],
        "Season": [2, 4, 27, 11],
        "Eliminated Week (Rank Rule)": ["Survived", "Week 7", "Survived", "Survived"],
        "Eliminated Week (Percent Rule)": ["Survived", "Week 7", "Survived", "Survived"],
        "Eliminated Week (Judge Save)": ["Week 2", "Week 4", "Week 3", "Week 5"]
    }
    df = pd.DataFrame(data)

def parse_week(val, default_max=10):
    if "Survived" in str(val):
        return default_max 
    try:
        return int(str(val).split()[-1])
    except:
        return 0

# 假设最大周数为 10 (用于可视化 Survived)
MAX_WEEKS = 10 

# 确定 "历史真实存活周数" (根据赛季区分 Rank/Percent)
df['Actual_Weeks'] = df.apply(lambda x: 
                              parse_week(x['Eliminated Week (Rank Rule)'], MAX_WEEKS) if x['Season'] < 10 
                              else parse_week(x['Eliminated Week (Percent Rule)'], MAX_WEEKS), axis=1)

df['JudgeSave_Weeks'] = df['Eliminated Week (Judge Save)'].apply(lambda x: parse_week(x, MAX_WEEKS))

# 按照“分歧程度”排序
df['Diff'] = df['Actual_Weeks'] - df['JudgeSave_Weeks']
df = df.sort_values('Diff', ascending=False)

# ==========================================
# 2. 绘图 (美化版)
# ==========================================
# 设置更干净的背景风格
sns.set_style("white") 
fig, ax = plt.subplots(figsize=(12, 6)) #稍微加宽一点

# y轴位置
y_positions = range(len(df))
bar_height = 0.5

# --- 配色升级 ---
# 使用更有质感的深红和钢蓝
color_actual = '#E63946'  # 深红色
color_save = '#457B9D'    # 钢蓝色

# 画 "真实历史" 的长条 (背景)
ax.barh(y_positions, df['Actual_Weeks'], height=bar_height, 
        color=color_actual, alpha=0.85, label='Actual Historical Survival')

# 画 "评委拯救" 的短条 (前景)
# 注意：这里把高度设为一样，但 alpha 设为 1，覆盖在上面，效果更整洁
ax.barh(y_positions, df['JudgeSave_Weeks'], height=bar_height, 
        color=color_save, alpha=1.0, label='Projected Survival (with Judge Save)',
        edgecolor='white', linewidth=1.5) # 加个白边让层次更分明

# ==========================================
# 3. 细节调整与标注 (核心修改)
# ==========================================
ax.set_yticks(y_positions)
# 加大字体，加粗人名
ax.set_yticklabels(df['Name'], fontsize=13, fontweight='bold', color='#333333')

ax.set_xlabel("Competition Week", fontsize=12, fontweight='bold', color='#555555')
ax.set_title("Fate Divergence Timeline:\nHow 'Judge Save' Could Shorten the Survival of Controversial Contestants", 
             fontsize=15, fontweight='bold', pad=25, color='#222222')

# 设置X轴刻度
ax.set_xlim(0, MAX_WEEKS + 0.5)
xticks = range(1, MAX_WEEKS + 1)
xticklabels = [f"W{i}" if i < MAX_WEEKS else "Finals" for i in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=11, fontweight='bold')

# 移除上方和右方的边框线，更现代
sns.despine()

# 添加图例，放在合适的位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

# --- 添加白色注释 ---
for i, (idx, row) in enumerate(df.iterrows()):
    diff = row['Actual_Weeks'] - row['JudgeSave_Weeks']
    if diff > 0:
        # 计算注释文本的位置：放在红色条的末端往左一点点
        text_x_pos = row['Actual_Weeks'] - 0.2
        
        # 添加文本
        # color='white': 白色文字
        # ha='right': 右对齐，确保文字在条形内部
        ax.text(text_x_pos, i, f"-{diff} Weeks Cut", 
                va='center', ha='right', 
                color='white', fontweight='bold', fontsize=12)
        
        # 添加一条细的连接虚线，增加视觉引导 (黑色改为深灰色，不抢戏)
        ax.plot([row['JudgeSave_Weeks'], row['Actual_Weeks']-0.1], [i, i], 
                color='#555555', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)

plt.tight_layout()
# 增加 dpi 让图片更清晰
plt.savefig("fate_divergence_timeline_beautified.png", dpi=300, bbox_inches='tight')
print("美化版图表已生成：fate_divergence_timeline_beautified.png")
plt.show()