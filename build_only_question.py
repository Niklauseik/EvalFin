import pandas as pd
import os
import re

# **数据集路径**
input_file = "datasets/convfinqa_without_answer.csv"  # ✅ 直接基于 without_answer 版本修改
output_file = "datasets/convfinqa_only_question.csv"

# **读取数据**
df = pd.read_csv(input_file)

# **去掉 Conversations 里的问题，只保留 Question 部分**
def extract_final_question(query):
    """ 
    只保留 `query` 中的背景信息和最终 `Question: ...` 部分，去掉 `Conversations: q0, q1, q2...` 
    """
    if "Conversations:" in query:
        query = re.sub(r"Conversations:.*?Question:", "Question:", query, flags=re.DOTALL).strip()
    return query

df["query"] = df["query"].apply(extract_final_question)

# **保存新数据集**
df.to_csv(output_file, index=False)
print(f"✅ 生成 `only_question` 数据集，存储路径: {output_file}")
