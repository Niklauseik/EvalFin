import pandas as pd
import json
import os

# 读取数据集
data_path = "datasets/finqa_cot.csv"
df = pd.read_csv(data_path)

# 目标存储路径
output_dir = "datasets/finqa_jsonl"
os.makedirs(output_dir, exist_ok=True)

# 目标 JSONL 文件
jsonl_file_A = os.path.join(output_dir, "finqa_answer_first.jsonl")
jsonl_file_B = os.path.join(output_dir, "finqa_cot_first.jsonl")

# 确保 'query', 'answer', 'cot' 列存在
required_columns = {"query", "answer", "cot"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"❌ 数据集中缺少必要列: {required_columns - set(df.columns)}")

# 处理 JSONL 逻辑
def process_row(query, answer, cot, order="answer_first"):
    """
    生成 JSONL 格式：
    - order="answer_first"  -> 先给答案，再给推理过程（带衔接语句）
    - order="cot_first"  -> 先给推理过程，最后给答案
    """
    # 构造 messages 列表
    messages = [
        {"role": "system", "content": "You are a financial question-answering assistant. Answer the given question based on the provided financial context."},
        {"role": "user", "content": query}
    ]

    # 先答案再推理（增加衔接语句）
    if order == "answer_first":
        messages.append({"role": "assistant", "content": f"Answer: {answer}\n\nLet's go through the reasoning step by step:\n\n{cot}"})
    # 先推理再答案
    elif order == "cot_first":
        messages.append({"role": "assistant", "content": f"{cot}\n\nFinal Answer: {answer}"})
    else:
        raise ValueError("❌ 'order' 只能是 'answer_first' 或 'cot_first'")

    return {"messages": messages}

# 逐行处理
with open(jsonl_file_A, "w") as f_A, open(jsonl_file_B, "w") as f_B:
    for _, row in df.iterrows():
        query, answer, cot = row["query"], str(row["answer"]).strip(), row["cot"]
        
        # 生成 JSONL 格式
        data_A = process_row(query, answer, cot, order="answer_first")
        data_B = process_row(query, answer, cot, order="cot_first")

        # 写入 JSONL
        f_A.write(json.dumps(data_A) + "\n")
        f_B.write(json.dumps(data_B) + "\n")

print(f"✅ JSONL 生成完成！\n - 格式 A（先答案后推理 + 衔接语句）: {jsonl_file_A}\n - 格式 B（先推理后答案）: {jsonl_file_B}")
