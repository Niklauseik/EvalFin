import json
import random
import os

# 设置数据集路径
data_path = "datasets/balanced_3000.jsonl"
train_path = "datasets/balanced_3000_train.jsonl"
valid_path = "datasets/balanced_3000_valid.jsonl"

# 确保结果存储目录存在
os.makedirs("datasets", exist_ok=True)

# 读取数据
with open(data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 打乱数据（保证划分的随机性）
random.seed(42)  # 设置随机种子，确保可复现
random.shuffle(data)

# 划分 80% 训练集，20% 验证集
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
valid_data = data[split_idx:]

# 保存训练集
with open(train_path, "w", encoding="utf-8") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

# 保存验证集
with open(valid_path, "w", encoding="utf-8") as f:
    for entry in valid_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ 数据集已划分完成！\n训练集: {len(train_data)} 条, 验证集: {len(valid_data)} 条")
