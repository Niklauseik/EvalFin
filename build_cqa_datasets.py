import pandas as pd
import re

# 读取原始数据
file_path = "datasets/convfinqa.csv"
df = pd.read_csv(file_path)

# 构建字典：先存储所有 turn 的答案（包括 turn=0）
dialogue_answers = {}

for _, row in df.iterrows():
    dialogue_id = row["dialogue_id"]
    turn = row["turn"]
    answer = row["answer"]

    if dialogue_id not in dialogue_answers:
        dialogue_answers[dialogue_id] = {}
    dialogue_answers[dialogue_id][turn] = answer

# 过滤掉 turn=0（但答案已存入字典）
df = df[df["turn"] > 0]

# 生成 with_answer 数据集
df_with_answer = df.copy()
df_without_answer = df.copy()  # 保留 {answerX}，不删除

def replace_placeholder_with_answer(query, dialogue_id, turn):
    """用对应的 answer 替换 query 中的 {answerX}"""
    matches = re.findall(r"\{answer(\d+)\}", query)
    
    for match in matches:
        turn_id = int(match)  # 提取出 {answerX} 里面的 X
        if dialogue_id in dialogue_answers and turn_id in dialogue_answers[dialogue_id]:
            query = query.replace(f"{{answer{turn_id}}}", str(dialogue_answers[dialogue_id][turn_id]))
    
    return query

# 处理 with_answer 数据集
df_with_answer["query"] = df_with_answer.apply(
    lambda row: replace_placeholder_with_answer(row["query"], row["dialogue_id"], row["turn"]),
    axis=1
)

# without_answer 数据集不做额外处理，保留 `{answerX}`

# 保存新数据集
df_with_answer.to_csv("datasets/convfinqa_with_answer.csv", index=False)
df_without_answer.to_csv("datasets/convfinqa_without_answer.csv", index=False)

print("✅ 数据集构建完成！（with_answer 替换了 {answerX}，without_answer 保留了 {answerX}）")
