import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind

from analysis_utils import (
    DIMENSION_COLS,
    FIGURE_DIR,
    PROCESSED_DIR,
    REGRESSION_DIR,
    load_merged_dataset,
    set_paper_theme,
)


OUTPUT_FIG = str(FIGURE_DIR)
OUTPUT_TAB = str(REGRESSION_DIR)
OUTPUT_PROCESSED = str(PROCESSED_DIR)


def save_current_figure(stem):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIG, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_FIG, f"{stem}.pdf"), bbox_inches="tight")


def save_summary(model, filename):
    with open(os.path.join(OUTPUT_TAB, filename), "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())


def main():
    set_paper_theme()
    print("Loading merged analysis dataset")
    df = load_merged_dataset()
    print("Shape:", df.shape)

    dimension_cols = [c for c in DIMENSION_COLS if c in df.columns]

    analysis_cols = [
        "url",
        "question_title",
        "answer_text",
        "author",
        "engagement",
        "log_engagement",
        "upvotes",
        "comments",
        "shares",
        "views",
        "log_views",
        "ai_probability",
        "is_ai",
        "ai_label",
        "year",
        "text_len",
        "log_text_len",
        "score",
    ] + dimension_cols
    analysis_cols = [c for c in analysis_cols if c in df.columns]
    df[analysis_cols].to_csv(
        os.path.join(OUTPUT_PROCESSED, "final_analysis_dataset.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("Final analysis dataset saved.")

    print("\n===== Engagement vs Score =====")
    score_df = df[["engagement", "score"]].dropna()
    model = sm.OLS(score_df["engagement"], sm.add_constant(score_df["score"])).fit(cov_type="HC3")
    print(model.summary())
    save_summary(model, "engagement_vs_score.txt")

    print("\n===== Score by Year and AI/Human =====")
    yearly_score = df[df["is_ai"].isin([0, 1])][["is_ai", "score", "year"]].dropna()
    yearly_mean = yearly_score.groupby(["is_ai", "year"])["score"].mean().reset_index()
    yearly_mean.to_csv(os.path.join(OUTPUT_TAB, "score_by_year_ai_human.csv"), index=False)

    plt.figure(figsize=(7.2, 4.2))
    for name, group in yearly_mean.groupby("is_ai"):
        label = "AI-like" if name == 1 else "Human-like"
        plt.plot(group["year"], group["score"], marker="o", label=label)
    plt.xlabel("Year")
    plt.ylabel("Average quality score")
    plt.title("Average Quality Score by Year")
    plt.legend()
    save_current_figure("score_by_year_ai_human")
    plt.close()

    print("\n===== Text Length vs Score =====")
    length_score = df[["log_text_len", "score"]].dropna()
    model = sm.OLS(length_score["score"], sm.add_constant(length_score["log_text_len"])).fit(cov_type="HC3")
    print(model.summary())
    save_summary(model, "text_length_vs_score.txt")

    print("\n===== AI vs Human Score Comparison =====")
    score_compare = df[df["is_ai"].isin([0, 1])][["score", "is_ai"]].dropna()
    plt.figure(figsize=(7.2, 4.2))
    score_groups = [
        score_compare[score_compare["is_ai"] == 0]["score"].dropna(),
        score_compare[score_compare["is_ai"] == 1]["score"].dropna(),
    ]
    plt.boxplot(score_groups, tick_labels=["Human-like", "AI-like"], showfliers=False, patch_artist=True)
    plt.xlabel("Content type")
    plt.ylabel("Quality score")
    plt.title("Quality Score Distribution")
    save_current_figure("score_distribution_violin")
    plt.close()

    human_scores = score_compare[score_compare["is_ai"] == 0]["score"]
    ai_scores = score_compare[score_compare["is_ai"] == 1]["score"]
    t_stat, p_value = ttest_ind(ai_scores, human_scores, equal_var=False, nan_policy="omit")
    pd.DataFrame(
        [
            {
                "n_ai": len(ai_scores),
                "n_human": len(human_scores),
                "ai_mean": ai_scores.mean(),
                "human_mean": human_scores.mean(),
                "welch_t": t_stat,
                "p_value": p_value,
            }
        ]
    ).to_csv(os.path.join(OUTPUT_TAB, "score_group_ttest.csv"), index=False)
    print(f"Welch t = {t_stat:.3f}, p = {p_value:.5f}")

    print("\n===== Main Regression =====")
    reg_cols = ["log_engagement", "ai_probability", "score", "log_text_len", "year"]
    if "log_views" in df.columns:
        reg_cols.append("log_views")
    reg_df = df[reg_cols].dropna().copy()

    x_cols = ["ai_probability", "log_text_len", "year", "score"]
    if "log_views" in reg_df.columns:
        x_cols.append("log_views")
    model_main = sm.OLS(reg_df["log_engagement"], sm.add_constant(reg_df[x_cols])).fit(cov_type="HC3")
    print(model_main.summary())
    save_summary(model_main, "main_regression.txt")

    print("\n===== Interaction Regression =====")
    reg_df["interaction"] = reg_df["ai_probability"] * reg_df["score"]
    x_cols_interaction = ["ai_probability", "score", "interaction", "log_text_len", "year"]
    if "log_views" in reg_df.columns:
        x_cols_interaction.append("log_views")
    model_interaction = sm.OLS(
        reg_df["log_engagement"],
        sm.add_constant(reg_df[x_cols_interaction]),
    ).fit(cov_type="HC3")
    print(model_interaction.summary())
    save_summary(model_interaction, "interaction_regression.txt")

    print("\n===== Quality Bin Analysis =====")
    bin_df = df[df["is_ai"].isin([0, 1])][["engagement", "log_engagement", "score", "is_ai"]].dropna().copy()
    bin_df["quality_bin"] = pd.qcut(bin_df["score"], 5, duplicates="drop")
    quality_summary = (
        bin_df.groupby(["quality_bin", "is_ai"], observed=True)
        .agg(n=("engagement", "size"), mean_engagement=("engagement", "mean"))
        .reset_index()
    )
    quality_summary.to_csv(os.path.join(OUTPUT_TAB, "engagement_by_quality_bin_summary.csv"), index=False)

    plot_data = quality_summary.copy()
    plot_data["bin_order"] = plot_data.groupby("is_ai").cumcount()
    plt.figure(figsize=(7.2, 4.2))
    for is_ai, group in plot_data.groupby("is_ai"):
        label = "AI-like" if is_ai == 1 else "Human-like"
        plt.plot(group["bin_order"], group["mean_engagement"], marker="o", label=label)
        for _, row in group.iterrows():
            plt.text(row["bin_order"], row["mean_engagement"], f"n={int(row['n'])}", fontsize=7, ha="center")
    plt.xticks(sorted(plot_data["bin_order"].unique()), [f"Q{i + 1}" for i in sorted(plot_data["bin_order"].unique())])
    plt.xlabel("Quality quintile")
    plt.ylabel("Average engagement")
    plt.title("Engagement by Quality Level and Content Type")
    plt.legend()
    save_current_figure("engagement_by_quality_bin")
    plt.close()

    print("\n===== Engagement vs Score Scatter =====")
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(reg_df["score"], reg_df["log_engagement"], alpha=0.25, s=12, edgecolors="none")
    plt.xlabel("Quality score")
    plt.ylabel("log(1 + engagement)")
    plt.title("Engagement vs Quality Score")
    save_current_figure("engagement_vs_score_scatter")
    plt.close()

    print("\n===== Engagement Distribution by AI/Human =====")
    eng_df = df[df["is_ai"].isin([0, 1])][["log_engagement", "is_ai"]].dropna()
    plt.figure(figsize=(7.2, 4.2))
    engagement_groups = [
        eng_df[eng_df["is_ai"] == 0]["log_engagement"].dropna(),
        eng_df[eng_df["is_ai"] == 1]["log_engagement"].dropna(),
    ]
    plt.boxplot(engagement_groups, tick_labels=["Human-like", "AI-like"], showfliers=False, patch_artist=True)
    plt.xlabel("Content type")
    plt.ylabel("log(1 + engagement)")
    plt.title("Engagement Distribution")
    save_current_figure("engagement_distribution_violin")
    plt.close()

    print("\nMain analysis completed successfully.")


if __name__ == "__main__":
    main()
