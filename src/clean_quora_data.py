import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import Counter

# ========= 1. 读取原始数据 =========
input_file = "quora/data/processed/quora_answers_selenium.csv"
output_file = "quora/data/processed/quora_answers_cleaned.csv"

df = pd.read_csv(input_file)

print("原始行数:", len(df))


# ========= 2. 去重 =========
df = df.drop_duplicates(subset=["url", "answer_text", "author"])
print("去重后行数:", len(df))


# ========= 3. 时间处理（3y → 年份） =========
CURRENT_DATE = datetime.now()
CURRENT_YEAR = CURRENT_DATE.year
CURRENT_MONTH = CURRENT_DATE.month
CURRENT_DAY = CURRENT_DATE.day

month_map = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

def convert_time_to_year(t):

    if pd.isna(t):
        return np.nan

    s = str(t).lower().strip()

    # ===== 1. Feb 1, 2023 =====
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},\s*(\d{4})",
        s
    )
    if m:
        year = int(m.group(2))
        if 2000 <= year <= CURRENT_YEAR:
            return year

    # ===== 2. Jan 2023 =====
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})",
        s
    )
    if m:
        year = int(m.group(2))
        if 2000 <= year <= CURRENT_YEAR:
            return year

    # ===== 3. 3y / 3 years ago =====
    m = re.search(r"(\d+)\s*(y|year)", s)
    if m:
        return CURRENT_YEAR - int(m.group(1))

    # ===== 4. 10mo =====
    m = re.search(r"(\d+)\s*(mo|month)", s)
    if m:
        months = int(m.group(1))
        year = CURRENT_YEAR if months < CURRENT_MONTH else CURRENT_YEAR - 1
        return year

    # ===== 5. weeks =====
    if re.search(r"\d+\s*(w|week)", s):
        return CURRENT_YEAR

    # ===== 6. days =====
    if re.search(r"\d+\s*(d|day)", s):
        return CURRENT_YEAR

    # ===== 7. yesterday today =====
    if any(k in s for k in ["yesterday", "today", "just now"]):
        return CURRENT_YEAR

    # ===== 8. Jan 7 =====
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})",
        s
    )
    if m:

        month = month_map[m.group(1)]
        day = int(m.group(2))

        if (month < CURRENT_MONTH) or (month == CURRENT_MONTH and day <= CURRENT_DAY):
            return CURRENT_YEAR
        else:
            return CURRENT_YEAR - 1

    return np.nan


df["year"] = df["time"].apply(convert_time_to_year)
print("\n==== 2026 examples ====")
print(df.loc[df["year"]==2026, "time"].head(20))
# 防止未来年份
df.loc[df["year"] > CURRENT_YEAR, "year"] = np.nan

df = df.drop(columns=["time"])

# ========= 4. 数值列统一格式 =========
numeric_cols = ["upvotes", "comments", "views", "shares"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ========= 5. 填充缺失值 =========
df[numeric_cols] = df[numeric_cols].fillna(0)

text_cols = ["question_title", "answer_text", "author"]

for col in text_cols:
    df[col] = df[col].fillna("")


# ========= 6. 清理文本格式 =========
def clean_text(s):
    if pd.isna(s):
        return ""

    s = str(s)

    # 统一换行符
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 去掉零宽字符（经常导致奇怪显示）
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)

    # 把太多换行压缩：多行 -> 1行空格
    s = re.sub(r"\n+", " ", s)
    # 去掉简单 HTML tag
    s = re.sub(r"<[^>]+>", " ", s)

    # 把常见 HTML entity 简化（至少不会乱）
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    # 压缩空格
    s = re.sub(r"\s+", " ", s)

    return s.strip()

df["answer_text"] = df["answer_text"].apply(clean_text)



def is_bad_text(s):
    if not s:
        return True

    # 太短基本没意义（也容易让detector不稳定）
    if len(s) < 30:
        return True

    # 单字符占比过高（比如“哇哇哇...”）
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return True
    top_ratio = Counter(chars).most_common(1)[0][1] / len(chars)
    if top_ratio > 0.25:
        return True

    # 非字母数字比例过高（大量符号/乱码）
    alnum = sum(ch.isalnum() for ch in s)
    if alnum / max(len(s), 1) < 0.3:
        return True

    return False

df["bad_text"] = df["answer_text"].apply(is_bad_text)
print("bad_text count:", df["bad_text"].sum())

# ========= 7. 创建 Engagement 分数 =========
df["engagement"] = df["upvotes"] + 2 * df["comments"] + df["shares"]


# ========= 8. 最终检查 =========
print("\n数据结构:")
print(df.info())

print("\n统计信息:")
print(df.describe())

print("\n最终行数:", len(df))
print("year value counts (top):")
print(df["year"].value_counts(dropna=False).head(10))

print("bad_text examples:")
print(df.loc[df["bad_text"]==True, "answer_text"].head(3).tolist())
df["text_len"] = df["answer_text"].str.len()

# ========= 9. 保存清洗后数据 =========
df.to_csv(output_file, index=False)

print("\n清洗完成！已保存为:", output_file)