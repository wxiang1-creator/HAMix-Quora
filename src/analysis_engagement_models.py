import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.tools.sm_exceptions import ValueWarning

from analysis_utils import (
    DIMENSION_COLS,
    ENGAGEMENT_COMPONENTS,
    REGRESSION_DIR,
    TABLE_DIR,
    bootstrap_mean_ci,
    load_merged_dataset,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ValueWarning)


def star(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def tidy_model_result(model_name, result, key_terms):
    rows = []
    for term in key_terms:
        if term not in result.params.index:
            continue
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": result.params[term],
                "std_err": result.bse[term],
                "p_value": result.pvalues[term],
                "sig": star(result.pvalues[term]),
                "n": int(result.nobs),
                "aic": getattr(result, "aic", np.nan),
                "bic": getattr(result, "bic", np.nan),
                "r_squared": getattr(result, "rsquared", np.nan),
                "pseudo_r_squared": getattr(result, "prsquared", np.nan),
            }
        )
    return rows


def fit_extended_models(df):
    needed = [
        "engagement",
        "log_engagement",
        "has_engagement",
        "ai_probability",
        "log_text_len",
        "year",
        "score",
        "question_id",
        "author_id",
    ] + DIMENSION_COLS
    if "log_views" in df.columns:
        needed.append("log_views")

    reg = df[needed].dropna().copy()
    reg["year"] = reg["year"].astype(int).astype(str)

    exposure = " + log_views" if "log_views" in reg.columns else ""
    quality_terms = " + ".join(DIMENSION_COLS)

    # Question fixed effects absorb stable question/topic popularity and context.
    base_formula = (
        "log_engagement ~ ai_probability + log_text_len"
        f"{exposure} + {quality_terms} + C(year) + C(question_id)"
    )
    compact_formula = (
        "log_engagement ~ ai_probability + log_text_len"
        f"{exposure} + score + C(year)"
    )

    model_rows = []

    ols_hc3 = smf.ols(compact_formula, data=reg).fit(cov_type="HC3")
    model_rows.extend(
        tidy_model_result("OLS log engagement, HC3", ols_hc3, ["ai_probability", "score", "log_text_len", "log_views"])
    )

    ols_fe = smf.ols(base_formula, data=reg).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg["author_id"]},
    )
    model_rows.extend(
        tidy_model_result(
            "OLS log engagement, question FE, author-clustered SE",
            ols_fe,
            ["ai_probability", "log_text_len", "log_views"] + DIMENSION_COLS,
        )
    )

    poisson_formula = base_formula.replace("log_engagement", "engagement")
    poisson = smf.glm(
        poisson_formula,
        data=reg,
        family=sm.families.Poisson(),
    ).fit(cov_type="cluster", cov_kwds={"groups": reg["author_id"]})
    model_rows.extend(
        tidy_model_result(
            "Poisson count model, question FE, author-clustered SE",
            poisson,
            ["ai_probability", "log_text_len", "log_views"] + DIMENSION_COLS,
        )
    )

    negbin = smf.glm(
        poisson_formula,
        data=reg,
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit(cov_type="cluster", cov_kwds={"groups": reg["author_id"]})
    model_rows.extend(
        tidy_model_result(
            "Negative binomial count model, question FE, author-clustered SE",
            negbin,
            ["ai_probability", "log_text_len", "log_views"] + DIMENSION_COLS,
        )
    )

    # The binary hurdle stage can become singular with hundreds of question
    # fixed effects because many questions have no within-question variation in
    # zero vs positive engagement. Use the compact specification here and keep
    # question fixed effects in the main OLS/count robustness models above.
    hurdle_formula = compact_formula.replace("log_engagement", "has_engagement")
    hurdle_logit = smf.glm(
        hurdle_formula,
        data=reg,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3", maxiter=200)
    model_rows.extend(
        tidy_model_result(
            "Hurdle part 1: any engagement logit",
            hurdle_logit,
            ["ai_probability", "log_text_len", "log_views"] + DIMENSION_COLS,
        )
    )

    positive = reg[reg["engagement"] > 0].copy()
    hurdle_positive = smf.glm(
        poisson_formula,
        data=positive,
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit(cov_type="cluster", cov_kwds={"groups": positive["author_id"]})
    model_rows.extend(
        tidy_model_result(
            "Hurdle part 2: positive engagement negative binomial",
            hurdle_positive,
            ["ai_probability", "log_text_len", "log_views"] + DIMENSION_COLS,
        )
    )

    result_df = pd.DataFrame(model_rows)
    result_df.to_csv(REGRESSION_DIR / "extended_engagement_models.csv", index=False)

    with open(REGRESSION_DIR / "extended_engagement_models.txt", "w", encoding="utf-8") as f:
        f.write("OLS compact model\n")
        f.write(ols_hc3.summary().as_text())
        f.write("\n\nOLS with question fixed effects and author-clustered SE\n")
        f.write(ols_fe.summary().as_text())
        f.write("\n\nPoisson count model\n")
        f.write(poisson.summary().as_text())
        f.write("\n\nNegative binomial count model\n")
        f.write(negbin.summary().as_text())
        f.write("\n\nHurdle part 1 logit\n")
        f.write(hurdle_logit.summary().as_text())
        f.write("\n\nHurdle part 2 negative binomial among positive engagement\n")
        f.write(hurdle_positive.summary().as_text())

    return result_df


def fit_component_models(df):
    rows = []
    needed_base = ["ai_probability", "log_text_len", "year", "score", "author_id"]
    if "log_views" in df.columns:
        needed_base.append("log_views")

    exposure = " + log_views" if "log_views" in df.columns else ""
    formula_tail = f"ai_probability + log_text_len{exposure} + score + C(year)"

    for outcome in ENGAGEMENT_COMPONENTS:
        if outcome not in df.columns:
            continue
        temp = df[[outcome] + needed_base].dropna().copy()
        temp["year"] = temp["year"].astype(int).astype(str)
        model = smf.glm(
            f"{outcome} ~ {formula_tail}",
            data=temp,
            family=sm.families.NegativeBinomial(alpha=1.0),
        ).fit(cov_type="cluster", cov_kwds={"groups": temp["author_id"]})
        rows.extend(tidy_model_result(f"{outcome} negative binomial", model, ["ai_probability", "score", "log_text_len", "log_views"]))

    out = pd.DataFrame(rows)
    out.to_csv(REGRESSION_DIR / "engagement_component_models.csv", index=False)
    return out


def grouped_gap_tests(df):
    test_df = df[df["is_ai"].isin([0, 1])].copy()
    test_df = test_df.dropna(subset=["engagement", "text_len", "score", "is_ai"])

    specs = {
        "text_length_quintile": pd.qcut(test_df["text_len"], 5, duplicates="drop"),
        "quality_quintile": pd.qcut(test_df["score"], 5, duplicates="drop"),
    }

    rows = []
    for spec_name, bins in specs.items():
        test_df[spec_name] = bins
        for bin_value, group in test_df.groupby(spec_name, observed=True):
            ai = group[group["is_ai"] == 1]["engagement"].dropna()
            human = group[group["is_ai"] == 0]["engagement"].dropna()
            if len(ai) < 5 or len(human) < 5:
                continue
            diff = ai.mean() - human.mean()
            rng = np.random.default_rng(42)
            boot_diffs = []
            ai_values = ai.to_numpy()
            human_values = human.to_numpy()
            for _ in range(1000):
                boot_ai = rng.choice(ai_values, size=len(ai_values), replace=True).mean()
                boot_human = rng.choice(human_values, size=len(human_values), replace=True).mean()
                boot_diffs.append(boot_ai - boot_human)
            diff_low, diff_high = np.percentile(boot_diffs, [2.5, 97.5])
            t_stat, t_p = ttest_ind(ai, human, equal_var=False, nan_policy="omit")
            u_stat, u_p = mannwhitneyu(ai, human, alternative="two-sided")
            rows.append(
                {
                    "bin_type": spec_name,
                    "bin": str(bin_value),
                    "n_ai": len(ai),
                    "n_human": len(human),
                    "ai_mean": ai.mean(),
                    "human_mean": human.mean(),
                    "mean_diff_ai_minus_human": diff,
                    "bootstrap_diff_ci_low": diff_low,
                    "bootstrap_diff_ci_high": diff_high,
                    "welch_t": t_stat,
                    "welch_p": t_p,
                    "mannwhitney_u": u_stat,
                    "mannwhitney_p": u_p,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "grouped_engagement_gap_tests.csv", index=False)
    return out


def main():
    df = load_merged_dataset()
    model_table = fit_extended_models(df)
    component_table = fit_component_models(df)
    gap_table = grouped_gap_tests(df)
    print("Saved:", REGRESSION_DIR / "extended_engagement_models.csv")
    print("Saved:", REGRESSION_DIR / "engagement_component_models.csv")
    print("Saved:", TABLE_DIR / "grouped_engagement_gap_tests.csv")
    print(model_table.head(10))
    print(component_table.head(10))
    print(gap_table.head(10))


if __name__ == "__main__":
    main()
