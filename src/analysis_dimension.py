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
DATA_PATH = "quora/data/processed/merged.csv"
OUTPUT_FIG = "quora/data/outputs/figures"
OUTPUT_TAB = "quora/data/outputs/descriptive_tables"

os.makedirs(OUTPUT_FIG, exist_ok=True)
os.makedirs(OUTPUT_TAB, exist_ok=True)

# =========================
# 1. Load data（🔥 修复编码问题）
# =========================
def safe_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            print(f"Trying encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Failed to read CSV with all encodings")

df = safe_read_csv(DATA_PATH)

# =========================
# 2. Columns
# =========================
dimension_cols = [
    "clarity_score",
    "completeness_score",
    "relevance_score",
    "originality_score",
    "authenticity_score"
]

missing = [c for c in dimension_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing dimension columns: {missing}")

# 转数值
for col in dimension_cols + ["engagement", "ai_probability", "text_len", "year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 自动生成 is_ai
if "is_ai" not in df.columns:
    df["is_ai"] = np.where(
        df["ai_probability"] >= 0.8, 1,
        np.where(df["ai_probability"] <= 0.2, 0, np.nan)
    )

# 只保留确定标签
df = df.dropna(subset=["is_ai"])

# =========================
# 3. Mean Comparison
# =========================
print("\n===== Dimension Mean Comparison =====")

mean_table = df.groupby("is_ai")[dimension_cols].mean().T
mean_table.columns = ["Human-like", "AI-like"]

print(mean_table)
mean_table.to_csv(os.path.join(OUTPUT_TAB, "dimension_mean_by_ai_human.csv"))

# Plot
mean_table.plot(kind="bar", figsize=(10,6))
plt.title("Average Dimension Scores by Content Type")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "dimension_mean_bar.png"))
plt.show()

# =========================
# 4. T-tests
# =========================
print("\n===== T-tests for Each Dimension =====")

ttest_results = []

for col in dimension_cols:
    human = df[df["is_ai"] == 0][col].dropna()
    ai = df[df["is_ai"] == 1][col].dropna()

    if len(human) > 5 and len(ai) > 5:
        t_stat, p_value = ttest_ind(ai, human, equal_var=False)
    else:
        t_stat, p_value = np.nan, np.nan

    ttest_results.append({
        "dimension": col,
        "human_mean": human.mean(),
        "ai_mean": ai.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    })

ttest_df = pd.DataFrame(ttest_results)
print(ttest_df)
ttest_df.to_csv(os.path.join(OUTPUT_TAB, "dimension_ttests.csv"), index=False)

# =========================
# 5. Trends Over Time
# =========================
print("\n===== Dimension Trends Over Time =====")

for col in dimension_cols:
    temp = df.groupby(["year", "is_ai"])[col].mean().reset_index()

    plt.figure(figsize=(8,5))
    for name, group in temp.groupby("is_ai"):
        label = "AI-like" if name == 1 else "Human-like"
        plt.plot(group["year"], group[col], marker="o", label=label)

    plt.title(f"{col} Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIG, f"{col}_over_time.png"))
    plt.show()

# =========================
# 6. Dimension vs Engagement
# =========================
print("\n===== Dimension vs Engagement Regression =====")

corr_results = []

for col in dimension_cols:
    temp = df[[col, "engagement"]].dropna()

    if len(temp) < 10:
        continue

    X = sm.add_constant(temp[col])
    y = temp["engagement"]

    model = sm.OLS(y, X).fit()

    corr_results.append({
        "dimension": col,
        "coef": model.params[col],
        "p_value": model.pvalues[col],
        "r_squared": model.rsquared
    })

corr_df = pd.DataFrame(corr_results)
print(corr_df)
corr_df.to_csv(os.path.join(OUTPUT_TAB, "dimension_vs_engagement_regression.csv"), index=False)

# =========================
# 7. Multivariate Regression
# =========================
print("\n===== Separate Regressions for Each Dimension =====")

df["log_engagement"] = np.log1p(df["engagement"])

multi_results = []

for col in dimension_cols:
    temp = df[[col, "ai_probability", "text_len", "year", "log_engagement"]].dropna()

    if len(temp) < 10:
        continue

    X = temp[[col, "ai_probability", "text_len", "year"]]
    X = sm.add_constant(X)
    y = temp["log_engagement"]

    model = sm.OLS(y, X).fit()

    multi_results.append({
        "dimension": col,
        "coef": model.params[col],
        "std_err": model.bse[col],
        "p_value": model.pvalues[col],
        "r_squared": model.rsquared
    })

multi_df = pd.DataFrame(multi_results)
print(multi_df)
multi_df.to_csv(os.path.join(OUTPUT_TAB, "dimension_multivariate_regression.csv"), index=False)

print("\n✅ Dimension analysis completed successfully.")