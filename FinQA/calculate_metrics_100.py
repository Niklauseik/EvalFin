import pandas as pd
import re
import os
from sklearn.metrics import accuracy_score

# **文件路径**
results_dir = "results/finqa/answer_first_test"
results_file = os.path.join(results_dir, "qa_100_predictions.csv")

# **读取测试数据**
df = pd.read_csv(results_file)

# **提取模型预测的数值（支持负数）**
def extract_number(text):
    """ 提取 'Answer:' 后面的数字，支持负数，并去掉前导0，仅保留前两位进行匹配 """
    match = re.search(r"Answer:\s*(-?[\d.,]+)", str(text))
    if match:
        num = match.group(1)
        num = re.sub(r",", "", num)  # 去除 `,`
        num = re.sub(r"^(-?)0+", r"\1", num)  # 去除前导 `0`（但保留负号）
        return num[:2]  # 取前两位（包含负号）
    return None

# **处理真实答案 & 预测答案**
df["true_number"] = df["answer"].astype(str).apply(lambda x: re.sub(r",", "", x))  # 先去 `,`
df["true_number"] = df["true_number"].apply(lambda x: re.sub(r"^(-?)0+", r"\1", x))  # 去 `0`
df["true_number"] = df["true_number"].apply(lambda x: x[:2])  # 取前两位

df["predicted_number"] = df["prediction"].astype(str).apply(extract_number)

# **找出能计算的行**
df_valid = df.dropna(subset=["true_number", "predicted_number"])

# **确保只包含有效的数值**
df_valid["true_number"] = df_valid["true_number"].astype(str)
df_valid["predicted_number"] = df_valid["predicted_number"].astype(str)

# **计算 Accuracy**
accuracy = accuracy_score(df_valid["true_number"], df_valid["predicted_number"])

# **保存 Accuracy 结果**
metrics_file = os.path.join(results_dir, "qa_100_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")

print(f"✅ Accuracy 计算完成: {accuracy:.4f}")
