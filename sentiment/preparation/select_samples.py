import pandas as pd
import random

# 读取提取后的 CSV 文件
df = pd.read_csv('datasets/fpb_text.csv')

# 筛选 Negative 样本
negative_df = df[df['answer'] == 'negative']

# 随机选择 400 条 Negative 样本
if len(negative_df) >= 400:
    selected_negative_df = negative_df.sample(n=400, random_state=42)  # 使用 random_state 确保可重复性
else:
    print(f"警告：Negative 样本只有 {len(negative_df)} 条，无法生成 400 条。请检查数据或调整需求。")
    selected_negative_df = negative_df  # 如果样本不足，使用所有 Negative 样本

# 保存为新数据集
selected_negative_df.to_csv('datasets/negative_400.csv', index=False)

print(f"已随机选择 {len(selected_negative_df)} 条 Negative 样本，保存到 datasets/negative_400.csv")