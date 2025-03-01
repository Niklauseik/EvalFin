import openai
import pandas as pd
import os
from utils import config_manager

# 读取 OpenAI API Key
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# **Fine-tuned `GPT-4o-mini`（QA 任务，渐进微调后）**
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:answer-first-1:B4g7fE6Y"  # ✅ 最新微调版本

# 读取测试数据集
test_file = "datasets/finqa.csv"
df = pd.read_csv(test_file)

# 确保 `finqa_4o` 结果存储目录存在
results_dir = "results/finqa/answer_first_2200"
os.makedirs(results_dir, exist_ok=True)

# 初始化预测列表
predictions = []

# 遍历测试数据进行预测
for i, row in df.iterrows():
    query = row["query"]  # 读取测试文本

    try:
        # 发送 API 请求
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": query}]
        )

        # 解析完整的 Response
        predicted_response = response.choices[0].message.content.strip()
        predictions.append(predicted_response)

    except Exception as e:
        print(f"❌ 预测失败，错误: {e}")
        predictions.append("error")  # 失败时填充 "error"

# **将预测结果添加到 DataFrame**
df["prediction"] = predictions

# **保存预测结果 CSV**
results_file = os.path.join(results_dir, "qa_with_predictions.csv")
df.to_csv(results_file, index=False)
print(f"✅ 预测结果已保存至: {results_file}")
