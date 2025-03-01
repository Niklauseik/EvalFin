import json
import random

# **原始 JSONL 文件路径**
input_jsonl_path = "datasets/finqa_jsonl/finqa_answer_first.jsonl"

# **已使用数据（1200 条）的 JSONL 文件**
used_jsonl_path = "datasets/finqa_jsonl/finqa_answer_first_1200.jsonl"

# **输出新抽取的 1000 条 JSONL**
output_jsonl_path = "datasets/finqa_jsonl/finqa_answer_first_new_1000.jsonl"

# **读取原始数据**
with open(input_jsonl_path, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

# **读取已使用的 1200 条数据**
with open(used_jsonl_path, "r", encoding="utf-8") as f:
    used_data = {json.dumps(json.loads(line), sort_keys=True) for line in f}  # 用 `set` 存已用数据

# **筛选出未使用的数据**
unused_data = [entry for entry in all_data if json.dumps(entry, sort_keys=True) not in used_data]

# **确保数据足够**
num_samples = 1000
if len(unused_data) < num_samples:
    raise ValueError(f"❌ 剩余数据不足！仅剩 {len(unused_data)} 条。")

# **随机抽取 1000 条**
selected_data = random.sample(unused_data, num_samples)

# **保存新的 1000 条 JSONL**
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for entry in selected_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 成功从 {len(all_data)} 条数据中抽取 {num_samples} 条新数据，并保存至 {output_jsonl_path} 🎯")
