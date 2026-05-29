import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

from analysis_utils import DIMENSION_COLS, REGRESSION_DIR, TABLE_DIR, load_merged_dataset


MATCH_COLS = ["log_text_len", "score", "year"] + DIMENSION_COLS


def standardized_mean_difference(df, treatment_col, cols):
    rows = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    for col in cols:
        t = treated[col].dropna()
        c = control[col].dropna()
        pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        rows.append(
            {
                "variable": col,
                "treated_mean": t.mean(),
                "control_mean": c.mean(),
                "smd": (t.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def estimate_propensity_scores(df):
    formula = "is_ai ~ log_text_len + score + C(year)"
    for col in DIMENSION_COLS:
        formula += f" + {col}"
    model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
    out = df.copy()
    out["propensity_score"] = model.predict(out)
    with open(REGRESSION_DIR / "propensity_score_model.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    return out


def nearest_neighbor_match(df, caliper=0.05):
    treated = df[df["is_ai"] == 1].copy()
    control = df[df["is_ai"] == 0].copy()

    matched_rows = []
    used_control = set()

    for treated_idx, row in treated.sort_values("propensity_score").iterrows():
        available = control.loc[~control.index.isin(used_control)].copy()
        if available.empty:
            break

        same_year = available[available["year"] == row["year"]]
        pool = same_year if not same_year.empty else available

        distances = cdist(
            [[row["propensity_score"]]],
            pool[["propensity_score"]].to_numpy(),
            metric="euclidean",
        )[0]
        best_pos = int(np.argmin(distances))
        best_distance = distances[best_pos]
        if best_distance > caliper:
            continue

        control_idx = pool.index[best_pos]
        used_control.add(control_idx)
        matched_rows.append(
            {
                "pair_id": len(matched_rows) + 1,
                "treated_index": treated_idx,
                "control_index": control_idx,
                "distance": best_distance,
            }
        )

    pairs = pd.DataFrame(matched_rows)
    if pairs.empty:
        return pairs, pd.DataFrame()

    treated_matched = df.loc[pairs["treated_index"]].copy()
    treated_matched["pair_id"] = pairs["pair_id"].to_numpy()
    treated_matched["matched_role"] = "AI-like"

    control_matched = df.loc[pairs["control_index"]].copy()
    control_matched["pair_id"] = pairs["pair_id"].to_numpy()
    control_matched["matched_role"] = "Human-like"

    matched = pd.concat([treated_matched, control_matched], ignore_index=True)
    return pairs, matched


def summarize_matched_gap(matched):
    rows = []
    for outcome in ["engagement", "log_engagement", "upvotes", "comments", "shares"]:
        if outcome not in matched.columns:
            continue
        ai = matched[matched["matched_role"] == "AI-like"][outcome].dropna()
        human = matched[matched["matched_role"] == "Human-like"][outcome].dropna()
        if len(ai) == 0 or len(human) == 0:
            continue
        t_stat, p_value = ttest_ind(ai, human, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "outcome": outcome,
                "n_ai": len(ai),
                "n_human": len(human),
                "ai_mean": ai.mean(),
                "human_mean": human.mean(),
                "mean_diff_ai_minus_human": ai.mean() - human.mean(),
                "welch_t": t_stat,
                "welch_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def main():
    df = load_merged_dataset()
    analysis = df[df["is_ai"].isin([0, 1])].dropna(subset=["is_ai", "engagement"] + MATCH_COLS).copy()
    analysis["year"] = analysis["year"].astype(int)

    before_balance = standardized_mean_difference(analysis, "is_ai", MATCH_COLS)
    before_balance["sample"] = "before_matching"

    scored = estimate_propensity_scores(analysis)
    pairs, matched = nearest_neighbor_match(scored, caliper=0.05)

    pairs.to_csv(TABLE_DIR / "matching_pairs.csv", index=False)

    if matched.empty:
        print("No matched pairs found. Try a wider caliper.")
        return

    matched.to_csv(TABLE_DIR / "matched_analysis_dataset.csv", index=False)

    after_balance = standardized_mean_difference(matched, "is_ai", MATCH_COLS)
    after_balance["sample"] = "after_matching"
    balance = pd.concat([before_balance, after_balance], ignore_index=True)
    balance.to_csv(TABLE_DIR / "matching_balance_smd.csv", index=False)

    gap = summarize_matched_gap(matched)
    gap.to_csv(TABLE_DIR / "matched_engagement_gap.csv", index=False)

    print("Matched pairs:", len(pairs))
    print("Saved:", TABLE_DIR / "matching_balance_smd.csv")
    print("Saved:", TABLE_DIR / "matched_engagement_gap.csv")
    print(gap)


if __name__ == "__main__":
    main()
