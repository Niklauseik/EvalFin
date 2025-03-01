import openai
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# **Fine-tuned `GPT-4o`（balanced 数据集）**
fine_tuned_model = "ft:gpt-4o-2024-08-06:personal:balanced:B3kEd6za"  # ✅ `GPT-4o balanced` 训练的模型

# 读取测试数据集
test_file = "datasets/sentiment.csv"
df = pd.read_csv(test_file)

# 确保 `balanced_4o` 结果存储目录存在
results_dir = "datasets/results/sentiment/balanced_4o"
os.makedirs(results_dir, exist_ok=True)

# 初始化预测列表
predictions = []

# 遍历测试数据进行预测
for i, row in df.iterrows():
    prompt = row["query"]  # 读取测试文本
    true_label = row["answer"]  # 真实标签

    try:
        # 发送 API 请求
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": prompt}]
        )

        # 解析模型预测的情感类别
        predicted_label = response.choices[0].message.content.strip().lower()
        predictions.append(predicted_label)

    except Exception as e:
        print(f"❌ 预测失败，错误: {e}")
        predictions.append("error")  # 失败时填充 "error"

# **将预测结果添加到 DataFrame**
df["prediction"] = predictions

# **保存预测结果 CSV**
results_file = os.path.join(results_dir, "sentiment_with_predictions.csv")
df.to_csv(results_file, index=False)
print(f"✅ 预测结果已保存至: {results_file}")

# **计算 Metrics**
true_labels = df["answer"].str.lower()
pred_labels = df["prediction"].str.lower()

# 计算 Accuracy, Precision, Recall, F1-score
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# **保存 Metrics 结果**
metrics_file = os.path.join(results_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"✅ 评估 Metrics 已保存至: {metrics_file}")
