import json
import random

# 原始 JSONL 文件路径（请替换为你的实际文件路径）
input_jsonl_path = "datasets/finqa_jsonl/finqa_answer_first.jsonl"

# 输出 JSONL 文件路径（抽取后的 1200 条）
output_jsonl_path = "datasets/finqa_jsonl/finqa_answer_first_1200.jsonl"

# 读取 JSONL 文件
with open(input_jsonl_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 确保数据量足够
num_samples = 1200
if len(data) < num_samples:
    raise ValueError(f"数据量不足！原数据集仅有 {len(data)} 条。")

# 随机抽取 1200 条数据
selected_data = random.sample(data, num_samples)

# 保存为新的 JSONL 文件
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for entry in selected_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 已成功从 {len(data)} 条数据中随机抽取 {num_samples} 条，并保存至 {output_jsonl_path}")
