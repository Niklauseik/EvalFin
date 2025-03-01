import pandas as pd
import json

# 📌 彻底清理 query 中的无用指令
def clean_text(text):
    # 可能的无用前缀，确保所有任务指令都被移除
    prefixes = [
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. Text:",
        "Analyze the sentiment of this statement extracted from a financial news article.",
        "Provide your answer as either negative, positive, or neutral.",
        "Text:"
    ]
    
    # 逐个检查并移除前缀
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "").strip()
    
    return text

# 📌 定义 CSV 转 JSONL 转换函数
def convert_csv_to_jsonl(input_csv, output_jsonl, sample_size=None):
    df = pd.read_csv(input_csv)

    # 如果需要随机抽样
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)  # 设定随机种子保证可复现

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 解析数据，清理 `query`
            text = clean_text(row['query'])

            label = row['answer'].strip().lower()  # 确保标签格式一致

            json_line = {
                "messages": [
                    {"role": "system", "content": "You are a financial sentiment classifier."},
                    {"role": "user", "content": text},  # 只保留新闻文本
                    {"role": "assistant", "content": label}
                ]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    print(f"✅ {input_csv} 已转换为 JSONL 格式，保存到 {output_jsonl}")

# 🚀 1. 处理 balanced_3000.csv
convert_csv_to_jsonl('datasets/balanced_3000.csv', 'datasets/balanced_3000.jsonl')

# 🚀 2. 处理 unbalanced_3000.csv
convert_csv_to_jsonl('datasets/unbalanced_3000.csv', 'datasets/unbalanced_3000.jsonl')

# 🚀 3. 从 merged_sentiment_balanced.csv 随机抽取 10 条样本
convert_csv_to_jsonl('datasets/merged_sentiment_balanced.csv', 'datasets/sample_10.jsonl', sample_size=10)
