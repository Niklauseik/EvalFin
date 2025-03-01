import openai
import pandas as pd
import os

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# **Fine-tuned Model (GPT-4o-mini)**
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:answer-first:B44PSLIx"  # ✅ 你的微调模型 ID

# 读取完整测试数据集
test_file = "datasets/finqa.csv"
df = pd.read_csv(test_file)

# **随机抽取 100 条测试数据**
df_sampled = df.sample(n=100, random_state=42).reset_index(drop=True)

# **更新结果存储目录**
results_dir = "results/finqa/answer_first_test"
os.makedirs(results_dir, exist_ok=True)

# **初始化预测列表**
predictions = []

# **遍历抽样测试数据进行推理**
for i, row in df_sampled.iterrows():
    prompt = row["query"]  # 读取问题和上下文
    true_answer = row["answer"]  # 真实答案

    try:
        # **发送 API 请求**
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": prompt}]
        )

        # **获取模型返回的答案**
        predicted_answer = response.choices[0].message.content.strip()
        predictions.append(predicted_answer)

    except Exception as e:
        print(f"❌ 预测失败，错误: {e}")
        predictions.append("error")  # 失败时填充 "error"

# **将预测结果添加到 DataFrame**
df_sampled["prediction"] = predictions

# **保存预测结果 CSV**
results_file = os.path.join(results_dir, "qa_100_predictions.csv")
df_sampled.to_csv(results_file, index=False)
print(f"✅ 预测结果已保存至: {results_file}")

# **下一步**
print("🚀 请手动检查部分预测结果，以确保格式正确！")
