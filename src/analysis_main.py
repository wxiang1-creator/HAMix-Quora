import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
import numpy as np

# =========================
# 0. Setup
# =========================

# 以当前脚本所在位置为基准，避免路径混乱
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "merged.csv")
OUTPUT_FIG = os.path.join(PROJECT_ROOT, "data", "outputs", "figures")
OUTPUT_TAB = os.path.join(PROJECT_ROOT, "data", "outputs", "regression_tables")
OUTPUT_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(OUTPUT_FIG, exist_ok=True)
os.makedirs(OUTPUT_TAB, exist_ok=True)
os.makedirs(OUTPUT_PROCESSED, exist_ok=True)

# =========================
# 1. Load data
# =========================
print("Loading data from:", DATA_PATH)

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except:
    try:
        df = pd.read_csv(DATA_PATH, encoding="gbk")
    except:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Data loaded successfully.")
print("Shape:", df.shape)

# 如果还没有总分，就自动生成
dimension_cols = [
    "clarity_score",
    "completeness_score",
    "relevance_score",
    "originality_score",
    "authenticity_score"
]

if "score" not in df.columns and all(col in df.columns for col in dimension_cols):
    df["score"] = df[dimension_cols].mean(axis=1)

# 基础变量处理
for col in ["engagement", "ai_probability", "text_len", "year", "score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 自动生成 is_ai
if "is_ai" not in df.columns:
    df["is_ai"] = np.where(
        df["ai_probability"] >= 0.8, 1,
        np.where(df["ai_probability"] <= 0.2, 0, np.nan)
    )

df["log_engagement"] = np.log1p(df["engagement"])

# 保存一个最终干净分析版
analysis_cols = [
    "question_title", "answer_text", "engagement", "log_engagement",
    "ai_probability", "is_ai", "year", "text_len", "score"
] + [c for c in dimension_cols if c in df.columns]

analysis_cols = [c for c in analysis_cols if c in df.columns]

analysis_df = df[analysis_cols].copy()
analysis_df.to_csv(
    os.path.join(OUTPUT_PROCESSED, "final_analysis_dataset.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("Final analysis dataset saved.")

# =========================
# 2. Engagement vs Score
# =========================
print("\n===== Engagement vs Score =====")
df_clean = df[['engagement', 'score']].dropna()

X = sm.add_constant(df_clean['score'])
Y = df_clean['engagement']
model = sm.OLS(Y, X).fit()
print(model.summary())

with open(os.path.join(OUTPUT_TAB, "engagement_vs_score.txt"), "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

# =========================
# 3. Score by Year and AI/Human
# =========================
print("\n===== Score by Year and AI/Human =====")
df_clean = df[['is_ai', 'score', 'year']].dropna()

df_mean = df_clean.groupby(['is_ai','year'])['score'].mean().reset_index()
print(df_mean)

plt.figure(figsize=(8,5))
for name, group in df_mean.groupby('is_ai'):
    label = "AI-like" if name == 1 else "Human-like"
    plt.plot(group['year'], group['score'], marker='o', label=label)

plt.xlabel('Year')
plt.ylabel('Average Score')
plt.title('Average Score by Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "score_by_year_ai_human.png"))
plt.show()

# =========================
# 4. Text Length vs Score
# =========================
print("\n===== Text Length vs Score =====")
df_clean = df[['text_len', 'score']].dropna()

X = sm.add_constant(df_clean['text_len'])
Y = df_clean['score']
model = sm.OLS(Y, X).fit()
print(model.summary())

with open(os.path.join(OUTPUT_TAB, "text_length_vs_score.txt"), "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

# =========================
# 5. AI vs Human Score Distribution
# =========================
print("\n===== AI vs Human Score Comparison =====")
df_clean = df[['score','is_ai']].dropna()

plt.figure(figsize=(8,5))
sns.violinplot(x='is_ai', y='score', data=df_clean, inner='box')
plt.xticks([0, 1], ['Human-like', 'AI-like'])
plt.xlabel('Content Type')
plt.ylabel('Score')
plt.title('Score Distribution: AI-like vs Human-like')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "score_distribution_violin.png"))
plt.show()

human_scores = df_clean[df_clean['is_ai']==0]['score']
ai_scores = df_clean[df_clean['is_ai']==1]['score']

t_stat, p_value = ttest_ind(ai_scores, human_scores, nan_policy='omit')
print(f"t = {t_stat:.3f}, p = {p_value:.5f}")

# =========================
# 6. Main Regression
# =========================
print("\n===== Main Regression =====")
df_clean = df[['engagement', 'ai_probability', 'score', 'text_len', 'year']].dropna()

reg_df = df_clean.copy()
reg_df["log_engagement"] = np.log1p(reg_df["engagement"])

X = reg_df[["ai_probability", "text_len", "year", "score"]]
X = sm.add_constant(X)
y = reg_df["log_engagement"]

model_main = sm.OLS(y, X).fit()
print(model_main.summary())

with open(os.path.join(OUTPUT_TAB, "main_regression.txt"), "w", encoding="utf-8") as f:
    f.write(model_main.summary().as_text())

# =========================
# 7. Interaction Model
# =========================
print("\n===== Interaction Regression =====")
reg_df["interaction"] = reg_df["ai_probability"] * reg_df["score"]

X = reg_df[["ai_probability", "score", "interaction", "text_len", "year"]]
X = sm.add_constant(X)
y = reg_df["log_engagement"]

model_interaction = sm.OLS(y, X).fit()
print(model_interaction.summary())

with open(os.path.join(OUTPUT_TAB, "interaction_regression.txt"), "w", encoding="utf-8") as f:
    f.write(model_interaction.summary().as_text())

# =========================
# 8. Quality Bin Analysis
# =========================
print("\n===== Quality Bin Analysis =====")
df_bin = df[['engagement', 'score', 'is_ai']].dropna().copy()
df_bin["quality_bin"] = pd.qcut(df_bin["score"], 5, duplicates="drop")

df2 = df_bin.groupby(["quality_bin", "is_ai"])["engagement"].mean().unstack()

df2.plot(marker='o', figsize=(8,5))
plt.xlabel('Quality Bin')
plt.ylabel('Average Engagement')
plt.title('Engagement by Quality Level and Content Type')
plt.legend(["Human-like", "AI-like"])
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "engagement_by_quality_bin.png"))
plt.show()

# Engagement vs Score Scatter
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_clean, x='score', y='engagement', alpha=0.4)
plt.xlabel('Score')
plt.ylabel('Engagement')
plt.title('Engagement vs Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "engagement_vs_score_scatter.png"))
plt.show()

# Engagement vs Score Scatter
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_clean, x='score', y='engagement', alpha=0.4)
plt.xlabel('Score')
plt.ylabel('Engagement')
plt.title('Engagement vs Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "engagement_vs_score_scatter.png"))
plt.show()

# Engagement Distribution by AI/Human
df_eng = df[['engagement','is_ai']].dropna()

plt.figure(figsize=(8,5))
sns.violinplot(x='is_ai', y='engagement', data=df_eng, inner='box')
plt.xticks([0, 1], ['Human-like', 'AI-like'])
plt.xlabel('Content Type')
plt.ylabel('Engagement')
plt.title('Engagement Distribution: AI-like vs Human-like')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "engagement_distribution_violin.png"))
plt.show()
print("\nMain analysis completed successfully.")