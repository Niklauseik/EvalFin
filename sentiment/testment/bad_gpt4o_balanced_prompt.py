import openai
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import config_manager

# **1️⃣ 初始化 OpenAI API**
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

client = openai.OpenAI(api_key=api_key)

# **2️⃣ 替换为你的 Fine-tuned 模型**
fine_tuned_model = "ft:gpt-4o-2024-08-06:personal:balanced:B3kEd6za"

# **3️⃣ 读取优化后的测试数据集**
test_file = "datasets/sentiment_cleaned.csv"  # 👈 这里使用优化后的数据集
df = pd.read_csv(test_file)

# **4️⃣ 确保结果存储目录**
results_dir = "datasets/results/sentiment/balanced_4o_optimized"
os.makedirs(results_dir, exist_ok=True)

# **5️⃣ 任务描述 & 示例**
system_message = """
You are a financial sentiment classifier. Given a financial news snippet, classify its sentiment into one of three categories: **Positive, Negative, or Neutral**.

**Sentiment Categories**:
- **Positive**: Optimistic, confident, or strong financial performance.
- **Negative**: Concerned, uncertain, or weak financial performance.
- **Neutral**: Factual, descriptive, without strong positive or negative sentiment.

**Examples**:
1. **"Apple's revenue surged by 20% this quarter, exceeding market expectations."** → **Positive**
2. **"Tesla's stock plummeted after the earnings call revealed weaker-than-expected guidance."** → **Negative**
3. **"Tim Cook has been appointed as Apple's CEO."** → **Neutral**
"""

# **6️⃣ 预测 & 记录结果**
predictions = []

for i, row in df.iterrows():
    text = row["query"]  # **这里 "query" 仅包含优化后的文本，不含冗余问题**
    true_label = row["answer"].lower()

    try:
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"**Text:** {text}\n**Sentiment:**"}
            ]
        )

        predicted_label = response.choices[0].message.content.strip().lower()
        predictions.append(predicted_label)

    except Exception as e:
        print(f"❌ 预测失败，错误: {e}")
        predictions.append("error")

# **7️⃣ 结果存储**
df["prediction"] = predictions
results_file = os.path.join(results_dir, "sentiment_with_predictions.csv")
df.to_csv(results_file, index=False)
print(f"✅ 预测结果已保存至: {results_file}")

# **8️⃣ 计算 Metrics**
true_labels = df["answer"].str.lower()
pred_labels = df["prediction"].str.lower()

accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# **9️⃣ 保存 Metrics 结果**
metrics_file = os.path.join(results_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"✅ 评估 Metrics 已保存至: {metrics_file}")