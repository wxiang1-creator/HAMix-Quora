import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    FIGURE_DIR,
    PAPER_PALETTE,
    TABLE_DIR,
    load_merged_dataset,
    save_figure,
    set_paper_theme,
)


AI_THRESHOLD = 0.8
HUMAN_THRESHOLD = 0.2
INCOMPLETE_YEAR = 2026


def assign_threshold_label(df):
    out = df.copy()
    out["ai_label"] = np.where(
        out["ai_probability"] >= AI_THRESHOLD,
        "AI-like",
        np.where(out["ai_probability"] <= HUMAN_THRESHOLD, "Human-like", "Uncertain"),
    )
    return out


def plot_content_mix_by_year(df):
    counts = df.groupby(["year", "ai_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["Human-like", "AI-like", "Uncertain"], fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.to_csv(TABLE_DIR / "content_mix_share_by_year.csv")
    counts.to_csv(TABLE_DIR / "content_mix_count_by_year.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(shares))
    x = np.arange(len(shares.index))
    for label in ["Human-like", "AI-like", "Uncertain"]:
        ax.bar(
            x,
            shares[label],
            bottom=bottom,
            label=label,
            color=PAPER_PALETTE[label],
            width=0.64,
        )
        bottom += shares[label].to_numpy()

    totals = counts.sum(axis=1)
    for i, (year, total) in enumerate(totals.items()):
        ax.text(i, 1.025, f"N={int(total):,}", ha="center", va="bottom", fontsize=8)
        if int(year) == INCOMPLETE_YEAR:
            ax.text(i, -0.08, "incomplete", ha="center", va="top", fontsize=8, color="#6B7280")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in shares.index])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of answers")
    ax.set_xlabel("Posting year")
    ax.set_title("Content-Type Composition by Year")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, "content_mix_share_by_year")
    plt.close(fig)


def plot_threshold_sensitivity(df):
    thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
    rows = []
    for threshold in thresholds:
        temp = df.dropna(subset=["ai_probability", "year"]).copy()
        temp["classified_ai"] = temp["ai_probability"] >= threshold
        share = temp.groupby("year")["classified_ai"].mean()
        count = temp.groupby("year")["classified_ai"].size()
        for year, value in share.items():
            rows.append(
                {
                    "threshold": threshold,
                    "year": year,
                    "ai_share": value,
                    "n": int(count.loc[year]),
                    "incomplete_year": int(year) == INCOMPLETE_YEAR,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "ai_share_threshold_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for threshold, group in out.groupby("threshold"):
        linestyle = "-" if threshold == AI_THRESHOLD else "--"
        linewidth = 2.2 if threshold == AI_THRESHOLD else 1.0
        alpha = 1.0 if threshold == AI_THRESHOLD else 0.45
        ax.plot(
            group["year"],
            group["ai_share"],
            marker="o",
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f"p >= {threshold}",
        )

    ax.axvspan(INCOMPLETE_YEAR - 0.35, INCOMPLETE_YEAR + 0.35, color="#E5E7EB", alpha=0.45)
    ax.text(INCOMPLETE_YEAR, ax.get_ylim()[1] * 0.92, "2026 incomplete", ha="center", fontsize=8, color="#4B5563")
    ax.set_ylabel("AI-like share")
    ax.set_xlabel("Posting year")
    ax.set_title("AI-like Share Across Probability Thresholds")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, "ai_share_threshold_sensitivity")
    plt.close(fig)


def plot_engagement_distribution(df):
    analysis = df[df["ai_label"].isin(["AI-like", "Human-like"])].dropna(subset=["engagement"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})

    for label in ["Human-like", "AI-like"]:
        values = np.log1p(analysis.loc[analysis["ai_label"] == label, "engagement"].dropna())
        axes[0].hist(
            values,
            bins=45,
            density=True,
            alpha=0.45,
            label=label,
            color=PAPER_PALETTE[label],
        )
    axes[0].legend()
    axes[0].set_xlabel("log(1 + engagement)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Long-tail Engagement Distribution")

    box_values = [
        np.log1p(analysis.loc[analysis["ai_label"] == "Human-like", "engagement"].dropna()),
        np.log1p(analysis.loc[analysis["ai_label"] == "AI-like", "engagement"].dropna()),
    ]
    box = axes[1].boxplot(
        box_values,
        tick_labels=["Human-like", "AI-like"],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )
    for patch, label in zip(box["boxes"], ["Human-like", "AI-like"]):
        patch.set_facecolor(PAPER_PALETTE[label])
        patch.set_alpha(0.75)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("log(1 + engagement)")
    axes[1].set_title("Central Distribution")

    save_figure(fig, "engagement_distribution_log_with_box")
    plt.close(fig)


def grouped_ci_table(df, value_col, bin_col, label):
    rows = []
    for (bin_value, ai_label), group in df.groupby([bin_col, "ai_label"], observed=True):
        values = group[value_col].dropna()
        if values.empty:
            continue
        se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        ci = 1.96 * se if pd.notna(se) else np.nan
        rows.append(
            {
                "metric": label,
                "bin": str(bin_value),
                "ai_label": ai_label,
                "n": len(values),
                "mean": values.mean(),
                "ci_low": values.mean() - ci if pd.notna(ci) else np.nan,
                "ci_high": values.mean() + ci if pd.notna(ci) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_engagement_by_bins(df):
    analysis = df[df["ai_label"].isin(["AI-like", "Human-like"])].dropna(
        subset=["engagement", "text_len", "score"]
    ).copy()
    analysis["text_length_quintile"] = pd.qcut(analysis["text_len"], 5, duplicates="drop")
    analysis["quality_quintile"] = pd.qcut(analysis["score"], 5, duplicates="drop")

    length_table = grouped_ci_table(analysis, "engagement", "text_length_quintile", "text_length")
    quality_table = grouped_ci_table(analysis, "engagement", "quality_quintile", "quality")
    out = pd.concat([length_table, quality_table], ignore_index=True)
    out.to_csv(TABLE_DIR / "engagement_by_bin_with_ci.csv", index=False)

    for metric, title, stem in [
        ("text_length", "Engagement by Text-Length Quintile", "engagement_by_text_length_bin_ci"),
        ("quality", "Engagement by Quality Quintile", "engagement_by_quality_bin_ci"),
    ]:
        plot_df = out[out["metric"] == metric].copy()
        plot_df["bin_order"] = plot_df.groupby("ai_label").cumcount()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for ai_label, group in plot_df.groupby("ai_label"):
            ax.errorbar(
                group["bin_order"],
                group["mean"],
                yerr=[group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]],
                marker="o",
                capsize=3,
                linewidth=2,
                label=ai_label,
                color=PAPER_PALETTE[ai_label],
            )
            for _, row in group.iterrows():
                ax.text(row["bin_order"], row["mean"], f"n={int(row['n'])}", fontsize=7, ha="center", va="bottom")
        ax.set_xticks(sorted(plot_df["bin_order"].unique()))
        ax.set_xticklabels([f"Q{i + 1}" for i in sorted(plot_df["bin_order"].unique())])
        ax.set_xlabel("Quintile")
        ax.set_ylabel("Mean engagement")
        ax.set_title(title)
        ax.legend(loc="upper left")
        save_figure(fig, stem)
        plt.close(fig)


def main():
    set_paper_theme()
    df = assign_threshold_label(load_merged_dataset())
    plot_content_mix_by_year(df)
    plot_threshold_sensitivity(df)
    plot_engagement_distribution(df)
    plot_engagement_by_bins(df)
    print("Updated trend and distribution figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
