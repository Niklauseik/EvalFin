import pandas as pd
import os   

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "../../datasets")

# 读取 CSV 文件
file_path = os.path.join(data_dir, "merged_sentiment_balanced.csv")
df = pd.read_csv(file_path)

# 统计不同类别的数量
label_counts = df['answer'].value_counts()

# 打印结果
print(label_counts)
