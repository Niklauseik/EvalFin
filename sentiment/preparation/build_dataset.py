import pandas as pd
import random

# 读取合并后的平衡数据集
merged_df = pd.read_csv('datasets/merged_sentiment_balanced.csv')

# 1. 生成类别均衡数据集（3,000 条，约 1,000 条每类）
# 统计合并数据集中的类别分布
merged_counts = merged_df['answer'].value_counts()

# 确保每类至少有 1,000 条样本（如果不足，可调整抽样数量）
target_per_class = 1000
balanced_sample = pd.DataFrame()

for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment in merged_counts:
        class_df = merged_df[merged_df['answer'] == sentiment]
        if len(class_df) >= target_per_class:
            sampled_df = class_df.sample(n=target_per_class, random_state=42)  # 随机抽样，确保可重复
        else:
            sampled_df = class_df  # 如果样本不足，使用所有样本
        balanced_sample = pd.concat([balanced_sample, sampled_df])

# 如果总样本少于 3,000 条，从多余的类别中补齐
while len(balanced_sample) < 3000:
    for sentiment in ['positive', 'negative', 'neutral']:
        if len(balanced_sample) >= 3000:
            break
        class_df = merged_df[merged_df['answer'] == sentiment]
        remaining_needed = 3000 - len(balanced_sample)
        if len(class_df) > remaining_needed:
            additional_df = class_df.sample(n=remaining_needed, random_state=42)
        else:
            additional_df = class_df
        balanced_sample = pd.concat([balanced_sample, additional_df]).drop_duplicates()

# 随机打乱顺序，确保样本随机性
balanced_sample = balanced_sample.sample(frac=1, random_state=42).iloc[:3000]

# 保存均衡数据集
balanced_sample.to_csv('datasets/balanced_3000.csv', index=False)
print("类别均衡数据集（3,000 条）已保存到 datasets/balanced_3000.csv")
print("均衡数据集类别分布：")
print(balanced_sample['answer'].value_counts())

# 2. 生成类别不均衡数据集（3,000 条，保持 merged_sentiment_balanced.csv 的原始分布）
# 统计合并数据集中的类别分布
merged_counts = merged_df['answer'].value_counts()
total_samples = 3000

# 根据合并数据集的原始分布比例抽样
unbalanced_sample = pd.DataFrame()
remaining_samples = total_samples

for sentiment in ['neutral', 'positive', 'negative']:  # 按类别顺序抽样
    if sentiment in merged_counts:
        proportion = merged_counts[sentiment] / merged_counts.sum()
        samples_needed = int(total_samples * proportion)
        if samples_needed > len(merged_df[merged_df['answer'] == sentiment]):
            samples_needed = len(merged_df[merged_df['answer'] == sentiment])
        class_df = merged_df[merged_df['answer'] == sentiment]
        sampled_df = class_df.sample(n=samples_needed, random_state=42)
        unbalanced_sample = pd.concat([unbalanced_sample, sampled_df])
        remaining_samples -= samples_needed

# 如果总样本少于 3,000 条，从剩余样本中补齐（按比例分配剩余样本）
if remaining_samples > 0:
    for sentiment in ['neutral', 'positive', 'negative']:
        if remaining_samples <= 0:
            break
        class_df = merged_df[merged_df['answer'] == sentiment]
        additional_needed = min(remaining_samples, len(class_df))
        if additional_needed > 0:
            additional_df = class_df.sample(n=additional_needed, random_state=42)
            unbalanced_sample = pd.concat([unbalanced_sample, additional_df])
            remaining_samples -= additional_needed

# 随机打乱顺序，确保样本随机性
unbalanced_sample = unbalanced_sample.sample(frac=1, random_state=42).iloc[:3000]

# 保存不均衡数据集
unbalanced_sample.to_csv('datasets/unbalanced_3000.csv', index=False)
print("类别不均衡数据集（3,000 条）已保存到 datasets/unbalanced_3000.csv")
print("不均衡数据集类别分布：")
print(unbalanced_sample['answer'].value_counts())