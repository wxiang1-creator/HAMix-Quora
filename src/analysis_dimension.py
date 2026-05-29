import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind

from analysis_utils import (
    DIMENSION_COLS,
    FIGURE_DIR,
    PAPER_PALETTE,
    TABLE_DIR,
    load_merged_dataset,
    save_figure,
    set_paper_theme,
)


def dimension_mean_tests(df):
    rows = []
    analysis = df[df["is_ai"].isin([0, 1])].copy()
    for col in DIMENSION_COLS:
        human = analysis[analysis["is_ai"] == 0][col].dropna()
        ai = analysis[analysis["is_ai"] == 1][col].dropna()
        t_stat, p_value = ttest_ind(ai, human, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "dimension": col,
                "n_ai": len(ai),
                "n_human": len(human),
                "human_mean": human.mean(),
                "ai_mean": ai.mean(),
                "mean_diff_ai_minus_human": ai.mean() - human.mean(),
                "welch_t": t_stat,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "dimension_ttests.csv", index=False)
    return out


def plot_dimension_means(test_table):
    plot_df = test_table.copy()
    plot_df["dimension"] = (
        plot_df["dimension"]
        .str.replace("_score", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )
    long_df = plot_df.melt(
        id_vars="dimension",
        value_vars=["human_mean", "ai_mean"],
        var_name="group",
        value_name="mean_score",
    )
    long_df["group"] = long_df["group"].map({"human_mean": "Human-like", "ai_mean": "AI-like"})

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    dimensions = list(plot_df["dimension"])
    x = range(len(dimensions))
    width = 0.36
    human = plot_df["human_mean"].to_numpy()
    ai = plot_df["ai_mean"].to_numpy()
    ax.bar([i - width / 2 for i in x], human, width=width, label="Human-like", color=PAPER_PALETTE["Human-like"])
    ax.bar([i + width / 2 for i in x], ai, width=width, label="AI-like", color=PAPER_PALETTE["AI-like"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(dimensions)
    ax.set_xlabel("")
    ax.set_ylabel("Mean score")
    ax.set_title("Average Quality-Dimension Scores by Content Type")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    save_figure(fig, "dimension_mean_bar")
    plt.close(fig)


def plot_dimension_trends(df):
    analysis = df[df["is_ai"].isin([0, 1])].copy()
    for col in DIMENSION_COLS:
        temp = (
            analysis.groupby(["year", "ai_label"])[col]
            .agg(mean="mean", n="size", se="sem")
            .reset_index()
        )
        temp["ci"] = 1.96 * temp["se"]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for label, group in temp.groupby("ai_label"):
            ax.errorbar(
                group["year"],
                group["mean"],
                yerr=group["ci"],
                marker="o",
                capsize=3,
                linewidth=2,
                label=label,
                color=PAPER_PALETTE[label],
            )
        ax.axvspan(2025.65, 2026.35, color="#E5E7EB", alpha=0.45)
        ax.set_title(f"{col.replace('_score', '').replace('_', ' ').title()} Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean score")
        ax.legend()
        save_figure(fig, f"{col}_over_time")
        plt.close(fig)


def multivariate_dimension_regression(df):
    cols = ["log_engagement", "ai_probability", "log_text_len", "year", "score"] + DIMENSION_COLS
    if "log_views" in df.columns:
        cols.append("log_views")
    reg = df[cols].dropna().copy()

    x_cols = ["ai_probability", "log_text_len", "year"] + DIMENSION_COLS
    if "log_views" in reg.columns:
        x_cols.append("log_views")

    model = sm.OLS(reg["log_engagement"], sm.add_constant(reg[x_cols])).fit(cov_type="HC3")
    rows = []
    for term in x_cols:
        rows.append(
            {
                "term": term,
                "coef": model.params[term],
                "std_err": model.bse[term],
                "p_value": model.pvalues[term],
                "n": int(model.nobs),
                "adj_r_squared": model.rsquared_adj,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "dimension_multivariate_regression.csv", index=False)

    with open(TABLE_DIR / "dimension_multivariate_regression.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    return out


def main():
    set_paper_theme()
    df = load_merged_dataset()
    test_table = dimension_mean_tests(df)
    plot_dimension_means(test_table)
    plot_dimension_trends(df)
    reg_table = multivariate_dimension_regression(df)
    print("Saved dimension tests and figures to:", TABLE_DIR, FIGURE_DIR)
    print(test_table)
    print(reg_table)


if __name__ == "__main__":
    main()
