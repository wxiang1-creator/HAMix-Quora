import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "descriptive_tables"
REGRESSION_DIR = OUTPUT_DIR / "regression_tables"

for path in [FIGURE_DIR, TABLE_DIR, REGRESSION_DIR]:
    os.makedirs(path, exist_ok=True)


DIMENSION_COLS = [
    "clarity_score",
    "completeness_score",
    "relevance_score",
    "originality_score",
    "authenticity_score",
]

ENGAGEMENT_COMPONENTS = ["upvotes", "comments", "shares"]

PAPER_PALETTE = {
    "Human": "#2F6F9F",
    "Human-like": "#2F6F9F",
    0.0: "#2F6F9F",
    0: "#2F6F9F",
    "AI": "#C75D2C",
    "AI-like": "#C75D2C",
    1.0: "#C75D2C",
    1: "#C75D2C",
    "Uncertain": "#8A8F98",
}


def set_paper_theme():
    rc = {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "grid.color": "#D9DEE5",
        "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
    }
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", rc=rc)
    else:
        import matplotlib.pyplot as plt

        plt.rcParams.update(rc)


def safe_read_csv(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Failed to read CSV with supported encodings: {path}")


def load_merged_dataset():
    df = safe_read_csv(PROCESSED_DIR / "merged.csv")
    return prepare_analysis_frame(df)


def prepare_analysis_frame(df):
    df = df.copy()

    numeric_cols = [
        "engagement",
        "ai_probability",
        "text_len",
        "year",
        "views",
        "upvotes",
        "comments",
        "shares",
    ] + DIMENSION_COLS

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "score" not in df.columns and all(col in df.columns for col in DIMENSION_COLS):
        df["score"] = df[DIMENSION_COLS].mean(axis=1)
    elif "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    elif "quality_score" in df.columns:
        df["score"] = pd.to_numeric(df["quality_score"], errors="coerce")

    if "is_ai" not in df.columns:
        df["is_ai"] = np.where(
            df["ai_probability"] >= 0.8,
            1,
            np.where(df["ai_probability"] <= 0.2, 0, np.nan),
        )
    else:
        df["is_ai"] = pd.to_numeric(df["is_ai"], errors="coerce")

    df["ai_label"] = np.where(
        df["is_ai"] == 1,
        "AI-like",
        np.where(df["is_ai"] == 0, "Human-like", "Uncertain"),
    )

    if "engagement" in df.columns:
        df["log_engagement"] = np.log1p(df["engagement"])
        df["has_engagement"] = (df["engagement"] > 0).astype(int)
    if "text_len" in df.columns:
        df["log_text_len"] = np.log1p(df["text_len"])
    if "views" in df.columns:
        df["log_views"] = np.log1p(df["views"])

    if "url" in df.columns:
        df["question_id"] = df["url"].astype(str)
    elif "question_title" in df.columns:
        df["question_id"] = df["question_title"].astype(str)

    if "author" in df.columns:
        df["author_id"] = df["author"].fillna("unknown_author").astype(str)

    return df


def save_figure(fig, stem):
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def bootstrap_mean_ci(values, n_boot=1000, seed=42):
    values = pd.Series(values).dropna().to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)
