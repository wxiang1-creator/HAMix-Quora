import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import json
import os

# 设置代理（可在程序里临时覆盖系统环境变量）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

client = OpenAI(
    api_key="OPENAI_API_KEY"
)

def LLM_judge(text):
    prompt = """
You are an expert in NLP and answer evaluation.

Task: 
Evaluate the quality of the Answer to the given Question and give scores across multiple dimensions.

Instructions:
Please evaluate the answer based on the following dimensions:

- 1. Clarity: 
Assess how easy the answer is to understand. Consider structure, logical flow, and whether the language is clear and unambiguous.

- 2. Completeness:
Assess whether the answer fully addresses the question. Consider coverage of key points and whether important details are missing.

- 3. Relevance:
Assess how closely the answer aligns with the question. Consider whether it stays on topic without unnecessary or off-topic content.

- 4. Originality:
Assess whether the answer provides non-generic, insightful content. Consider depth of reasoning and avoidance of boilerplate responses.

- 5. Authenticity:
Assess whether the answer feels natural and human-like. Consider tone, fluency, emotional expression and whether it avoids repetitive or overly mechanical phrasing.

Tips:
- Use scores from 0 to 1 in increments of 0.1
- Be strict but fair: small issues → slight deduction, major issues → larger penalty
- Each dimension should be evaluated and scored independently
- Do not infer information that is not present in the answer
- Focus only on the given Question and Answer

Now, please evaluate the following case and output the response in JSON format only. Do NOT include any extra text before or after JSON.

The JSON format should be:

{
    "clarity_score": ...,
    "completeness_score": ...,
    "relevance_score": ...,
    "originality_score": ...,
    "authenticity_score": ...
}
"""
    user_prompt = f'''Here is the texts:\n{text}\n'''
    response = client.chat.completions.create(
        model="gpt-5-mini",
        store=True,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt
             }
        ],
        stream=False,
        timeout=100,
    )
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    df = pd.read_csv(r'quora/data/processed/quora_answers_with_ai.csv')

    question_data = df['question_title'].tolist()
    answer_data = df['answer_text'].tolist()
    data = []
    for q, a in zip(question_data, answer_data):
        q_a = f'Question: {q}  Answer: {a}'
        data.append(q_a)
    # print(len(data))

    score_result = []
    n = 0
    for i in tqdm(data[9000:]):
        n += 1
        try:
            result = LLM_judge(i)
            json_result = json.loads(result)
            json_result['Q_A'] = i
            # print(json_result)

        except Exception as e:
            print(n, e)
            continue

        score_result.append(json_result)

    result_df = pd.DataFrame(score_result)
    result_df.to_csv("LLM_score_result_5.csv", index=False, encoding="utf-8-sig")
