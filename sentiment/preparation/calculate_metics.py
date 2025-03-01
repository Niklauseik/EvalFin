import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# **读取测试数据**
results_file = "datasets/results/sentiment/balanced_4o_optimized/sentiment_with_predictions.csv"
df = pd.read_csv(results_file)

# **转换为小写，确保一致**
df["answer"] = df["answer"].str.lower()
df["prediction"] = df["prediction"].str.lower()

# **计算 Accuracy, Precision, Recall, F1-score**
accuracy = accuracy_score(df["answer"], df["prediction"])
precision, recall, f1, _ = precision_recall_fscore_support(df["answer"], df["prediction"], average="macro", zero_division=0)

# **存储分析结果**
metrics_text = f"""Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
"""

# **保存到文件**
metrics_file = "datasets/results/sentiment/balanced_4o_optimized/metrics.txt"
with open(metrics_file, "w") as f:
    f.write(metrics_text)

# **打印结果**
print("✅ **Overall Metrics:**")
print(metrics_text)
print(f"\n📁 Metrics saved to: {metrics_file}")
