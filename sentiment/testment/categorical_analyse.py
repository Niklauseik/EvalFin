import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取测试结果文件
results_file = "datasets/results/sentiment/balanced_4o/sentiment_with_predictions.csv"
df = pd.read_csv(results_file)

# 转换为小写，确保一致性
df["answer"] = df["answer"].str.lower()
df["prediction"] = df["prediction"].str.lower()

# 获取唯一类别
categories = df["answer"].unique()

# 初始化存储 metrics 结果的字典
metrics_dict = {}

# 计算整体 Accuracy
overall_accuracy = accuracy_score(df["answer"], df["prediction"])

# 按类别计算 Precision, Recall, F1
for category in categories:
    true_labels = (df["answer"] == category).astype(int)
    pred_labels = (df["prediction"] == category).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    metrics_dict[category] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# 保存按类别计算的 Metrics
metrics_file = "datasets/results/sentiment/balanced_4o/category_metrics.txt"
with open(metrics_file, "w") as f:
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
    for category, metrics in metrics_dict.items():
        f.write(f"Category: {category.capitalize()}\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['F1 Score']:.4f}\n")
        f.write("\n")

print(f"✅ 按类别计算的 Metrics 已保存至: {metrics_file}")
