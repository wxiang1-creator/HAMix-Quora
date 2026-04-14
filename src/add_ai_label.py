import os
import time
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ========= 1. 读数据 =========
input_file = "quora/data/processed/quora_answers_cleaned.csv"
output_file = "quora/data/processed/quora_answers_with_ai.csv"

df = pd.read_csv(input_file)

# 只保留 2023+ （注意 year 里有 NaN）
df = df[df["year"].fillna(0) >= 2023].copy()

# 过滤 bad_text / 太短文本（强烈建议）
if "bad_text" in df.columns:
    df = df[df["bad_text"] == False].copy()

df["answer_text"] = df["answer_text"].fillna("").astype(str)
df = df[df["answer_text"].str.len() >= 300].copy()

print("样本数:", len(df))

# ========= 2. Winston API =========
API_KEY = "Winston_API_KEY"
url = "https://api.gowinston.ai/v2/ai-content-detection"

session = requests.Session()

def detect_ai(text, max_retry=3, timeout=30):
    if pd.isna(text):
        return np.nan

    text = str(text)

    if len(text) < 300:
        return np.nan

    payload = {"text": text[:1200]}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retry):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=timeout)

            # 成功
            if r.status_code == 200:
                data = r.json()
                score = data.get("score", np.nan)

                if score is None:
                    return np.nan

                score = float(score)
                # Winston 常见返回 0-100，统一到 0-1
                if score > 1:
                    score = score / 100.0
                        # 转换成 AI probability
                ai_score = 1 - score
                return ai_score

            # 速率限制：等一下再试
            if r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            # 其他错误：打印一下（只打印前几次避免刷屏）
            if attempt == 0:
                print(f"[WARN] status={r.status_code}, body={r.text[:200]}")
            time.sleep(1)

        except Exception as e:
            if attempt == 0:
                print(f"[WARN] request error: {e}")
            time.sleep(1)

    return np.nan


print("开始调用 Winston API...")
ai_scores = []

for text in tqdm(df["answer_text"], total=len(df)):
    score = detect_ai(text)
    ai_scores.append(score)
    time.sleep(0.15)  # 防止限速

df["ai_probability"] = np.round(ai_scores, 3)

# ========= 3. 高置信分组 =========
q80 = 0.8
q20 = 0.2

df["is_ai"] = np.where(
    df["ai_probability"] >= q80, 1,
    np.where(df["ai_probability"] <= q20, 0, np.nan)
)

print("高置信 AI 数量:")
print(df["is_ai"].value_counts(dropna=False))

print("\nai_probability 缺失率:", df["ai_probability"].isna().mean())

df.to_csv(output_file, index=False)
print("已保存:", output_file) 