import os
import pandas as pd
import matplotlib.pyplot as plt
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
# 1. Safe CSV Load
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
# 2. Basic Cleaning
# =========================
for col in ["ai_probability", "engagement", "text_len", "year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["ai_probability", "engagement", "text_len", "year"]).copy()

# =========================
# 3. Threshold Classification
# =========================
AI_THRESHOLD = 0.8
HUMAN_THRESHOLD = 0.2

df["ai_label"] = np.where(
    df["ai_probability"] >= AI_THRESHOLD,
    "AI",
    np.where(df["ai_probability"] <= HUMAN_THRESHOLD, "Human", "Uncertain")
)

print("\n===== Classification Result =====")
print(df["ai_label"].value_counts())

df_analysis = df[df["ai_label"] != "Uncertain"].copy()

print("\n===== Sample Used for Analysis =====")
print("N =", len(df_analysis))

# Save class counts
df["ai_label"].value_counts().to_csv(
    os.path.join(OUTPUT_TAB, "ai_label_counts.csv")
)

# =========================
# 4. AI / Human Count by Year
# =========================
count_by_year = df_analysis.groupby(["year", "ai_label"]).size().unstack(fill_value=0)

print("\n===== AI / Human Count by Year =====")
print(count_by_year)

count_by_year.to_csv(os.path.join(OUTPUT_TAB, "count_by_year_ai_human.csv"))

# Plot
count_by_year.plot(kind="bar", figsize=(10,6))
plt.title("AI vs Human Content Count by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "count_by_year_ai_human.png"))
plt.show()

# =========================
# 5. AI Proportion by Year
# =========================
ratio_by_year = count_by_year.div(count_by_year.sum(axis=1), axis=0)

print("\n===== AI Proportion by Year =====")
print(ratio_by_year["AI"])

ratio_by_year.to_csv(os.path.join(OUTPUT_TAB, "ai_ratio_by_year.csv"))

ratio_by_year["AI"].plot(marker='o', figsize=(8,5))
plt.title("AI Proportion Over Time (Threshold 0.2 / 0.8)")
plt.xlabel("Year")
plt.ylabel("AI Share")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "ai_proportion_over_time.png"))
plt.show()

# =========================
# 6. Average AI Probability by Year
# =========================
avg_prob = df.groupby("year")["ai_probability"].mean()

print("\n===== Average AI Probability by Year =====")
print(avg_prob)

avg_prob.to_csv(os.path.join(OUTPUT_TAB, "avg_ai_probability_by_year.csv"))

avg_prob.plot(marker='o', figsize=(8,5))
plt.title("Average AI Probability Over Time")
plt.xlabel("Year")
plt.ylabel("Average AI Probability")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "avg_ai_probability_over_time.png"))
plt.show()

# =========================
# 7. Robustness Test
# =========================
thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
robustness_results = []

plt.figure(figsize=(8,5))

for t in thresholds:
    df["ai_temp"] = np.where(df["ai_probability"] >= t, "AI", "Human")
    temp = df.groupby(["year", "ai_temp"]).size().unstack(fill_value=0)
    ratio = temp["AI"] / temp.sum(axis=1)

    for year, value in ratio.items():
        robustness_results.append({
            "threshold": t,
            "year": year,
            "ai_share": value
        })

    plt.plot(ratio.index, ratio.values, marker='o', label=f"t={t}")

plt.title("AI Proportion Under Different Thresholds")
plt.xlabel("Year")
plt.ylabel("AI Share")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "robustness_threshold_test.png"))
plt.show()

robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv(os.path.join(OUTPUT_TAB, "robustness_threshold_results.csv"), index=False)

# =========================
# 8. Average Text Length (AI vs Human)
# =========================
print("\n===== Average Text Length: AI vs Human =====")

avg_len = df_analysis.groupby("ai_label")["text_len"].mean()
print(avg_len)

avg_len.to_csv(os.path.join(OUTPUT_TAB, "avg_text_length_ai_vs_human.csv"))

avg_len.plot(kind='bar', figsize=(6,5))
plt.title("Average Text Length (AI vs Human)")
plt.ylabel("Average Text Length")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "avg_text_length_ai_vs_human.png"))
plt.show()

# =========================
# 9. Text Length Over Time
# =========================
len_by_year = df_analysis.groupby(["year", "ai_label"])["text_len"].mean().unstack()

print("\n===== Text Length Over Time =====")
print(len_by_year)

len_by_year.to_csv(os.path.join(OUTPUT_TAB, "text_length_by_year.csv"))

len_by_year.plot(marker='o', figsize=(8,5))
plt.title("Text Length Over Time (AI vs Human)")
plt.xlabel("Year")
plt.ylabel("Average Text Length")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "text_length_over_time.png"))
plt.show()

# =========================
# 10. AI-weighted Engagement
# =========================
df_analysis["ai_engagement"] = df_analysis["ai_probability"] * df_analysis["engagement"]

ae_by_year = df_analysis.groupby("year")["ai_engagement"].mean()

print("\n===== AI-weighted Engagement by Year =====")
print(ae_by_year)

ae_by_year.to_csv(os.path.join(OUTPUT_TAB, "ai_weighted_engagement_by_year.csv"))

ae_by_year.plot(marker='o', figsize=(8,5))
plt.title("AI Influence (AI Probability × Engagement)")
plt.xlabel("Year")
plt.ylabel("AI-weighted Engagement")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "ai_weighted_engagement.png"))
plt.show()

# =========================
# 11. Engagement Distribution (Long Tail)
# =========================
plt.figure(figsize=(8,5))

for label in ["AI", "Human"]:
    subset = df_analysis[df_analysis["ai_label"] == label]["engagement"].dropna()
    plt.hist(
        np.log1p(subset),
        bins=50,
        alpha=0.5,
        label=label
    )

plt.title("Engagement Distribution (log scale)")
plt.xlabel("log(1 + engagement)")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "engagement_distribution_log.png"))
plt.show()

# =========================
# 12. Continuous AI Presence
# =========================
ai_cont = df.groupby("year")["ai_probability"].mean()

ai_cont.to_csv(os.path.join(OUTPUT_TAB, "continuous_ai_presence_by_year.csv"))

ai_cont.plot(marker='o', figsize=(8,5))
plt.title("AI Presence Over Time (Continuous)")
plt.xlabel("Year")
plt.ylabel("Average AI Probability")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "continuous_ai_presence.png"))
plt.show()

# =========================
# 13. Text Length vs Engagement
# =========================
df_analysis["length_bin"] = pd.qcut(df_analysis["text_len"], 5, duplicates="drop")

len_eng = df_analysis.groupby("length_bin")["engagement"].mean()

print("\n===== Text Length vs Engagement =====")
print(len_eng)

len_eng.to_csv(os.path.join(OUTPUT_TAB, "text_length_vs_engagement.csv"))

len_eng.plot(kind='bar', figsize=(8,5))
plt.title("Engagement by Text Length")
plt.ylabel("Average Engagement")
plt.xlabel("Text Length Bin")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG, "text_length_vs_engagement.png"))
plt.show()

print("\n✅ analysis_trend.py completed successfully.")