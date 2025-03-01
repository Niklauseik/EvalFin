from datasets import load_dataset
import pandas as pd

# 加载数据集
ds = load_dataset("TheFinAI/Fino1_Reasoning_Path_FinQA")

# 提取 query 和 answer，并合并数据
df = pd.concat([
    pd.DataFrame(ds["train"])[["Open-ended Verifiable Question", "Ground-True Answer", "Complex_CoT"]],
])

# 保存到 CSV 文件
csv_path = "datasets/finqa_cot.csv"
df.to_csv(csv_path, index=False)

print(f"数据已保存至 {csv_path}")
