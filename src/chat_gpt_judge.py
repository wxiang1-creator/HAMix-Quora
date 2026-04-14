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

Task: Evaluate the quality of the Answer to the given Question.

Instructions:
1. Reasoning: 
- Explain briefly how well the Answer addresses the Question, considering relevance, correctness, completeness, and clarity.
- Avoid assumptions and base your evaluation only on the content provided.

2. Answer Quality Score (0 to 1, increments of 0.1)
0 = Completely irrelevant, no connection to the Question
0.1 = Almost irrelevant, barely related
0.2 = Very poor, minimal usefulness
0.3 = Poor, addresses Question weakly
0.4 = Weak, partial but incomplete answer
0.5 = Moderate, somewhat helpful, limited detail
0.6 = Fair, relevant and somewhat informative
0.7 = Good, clearly answers but not complete
0.8 = Very good, strong and clear answer
0.9 = Excellent, highly informative and relevant
1.0 = Perfect, fully correct, complete, and detailed

Now, please carefully evaluate the following case and output the response in JSON format only. Do NOT include any extra text before or after JSON.

The JSON format should be:
{
  "reasoning": "...",
  "score": ...
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
    print(len(data))

    score_result = []
    n = 0
    for i in tqdm(data[10000:]):
        n += 1
        try:
            result = LLM_judge(i)
            json_result = json.loads(result)
        except Exception as e:
            print(n, e)
            continue
        score_result.append(json_result)

    result_df = pd.DataFrame(score_result)
    result_df.to_csv("LLM_score_result_8.csv", index=False, encoding="utf-8-sig")
