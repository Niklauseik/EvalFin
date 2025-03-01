import pandas as pd

dataset_name = "fpb"
path = f"datasets/{dataset_name}.csv"
# 读取 CSV 文件
df = pd.read_csv(path)

# 提取 Text 部分
# 假设 query 格式为 "What is the sentiment of the following financial post: Positive, Negative, or Neutral? Text: [text]"
def extract_text(query):
    # 查找 "Text: " 后面的内容
    text_start = query.find("Text: ") + len("Text: ")
    if text_start != -1:  # 确保找到 "Text: "
        return query[text_start:].strip()
    return ""

# 应用函数提取 Text
df['text_only'] = df['query'].apply(extract_text)

# 保存提取后的数据（包含原有的 query 和 answer，以及新的 text_only 列）

df.to_csv(f"datasets/{dataset_name}_text.csv", index=False)

print(f"Text 提取完成，已保存到 datasets/{dataset_name}_text.csv")