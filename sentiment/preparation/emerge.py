import pandas as pd

# 步骤 1：为 negative_400_backtranslated.csv 添加 Prompt，生成 query 列（不保存中间文件）
# 读取 negative_400_backtranslated.csv
df_backtranslated = pd.read_csv('datasets/negative_400_backtranslated.csv')

# 定义 Prompt 模板
prompt_template = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text: "

# 为 back_translated_text 添加 Prompt，生成 query
df_backtranslated['query'] = df_backtranslated['back_translated_text'].apply(lambda x: prompt_template + str(x))

# 仅保留 query 和 answer 列，供后续合并使用（不保存到文件）
df_negative_400 = df_backtranslated[['query', 'answer']]

print("已为 negative_400_backtranslated.csv 添加 Prompt，准备合并")

# 步骤 2：读取 fpb.csv（原不均匀数据集）并合并
# 读取原始数据集 fpb.csv
df_original = pd.read_csv('datasets/fpb.csv')

# 合并数据集（垂直拼接，保留 query 和 answer 列）
merged_df = pd.concat([df_original, df_negative_400], ignore_index=True)

# 保存合并后的数据集
merged_df.to_csv('datasets/merged_sentiment_balanced.csv', index=False)

print("已将 negative_400 数据和 fpb.csv 合并，保存到 datasets/merged_sentiment_balanced.csv")

# 验证合并后的数据集（可选，打印类别分布和前 5 条数据）
sentiment_counts = merged_df['answer'].value_counts()
print("\n合并后数据集的类别分布：")
print(sentiment_counts)

print("\n前 5 条数据示例：")
print(merged_df.head())

# 确保格式正确（query 包含 Prompt，answer 为 negative/positive/neutral）
print("\n前 5 条数据的详细示例：")
for _, row in merged_df.head().iterrows():
    print(f"Query: {row['query']}")
    print(f"Answer: {row['answer']}")
    print("-" * 50)