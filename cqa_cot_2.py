import openai
import pandas as pd
import os
from utils import config_manager

# **读取 OpenAI API Key**
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# **初始化 OpenAI 客户端**
client = openai.OpenAI(api_key=api_key)

# **测试数据集路径**
datasets = {
    "without_answer_cot": "datasets/convfinqa_without_answer.csv",  # ✅ 直接使用 without_answer 数据集
}

# **确保结果存储目录存在**
results_dir = "results/cqa"
os.makedirs(results_dir, exist_ok=True)

# **遍历数据集进行推理**
for dataset_name, file_path in datasets.items():
    print(f"🚀 正在测试数据集: {dataset_name}（添加 CoT）")

    # **读取数据**
    df = pd.read_csv(file_path)

    # **初始化预测列表**
    predictions = []

    # **遍历测试数据进行预测**
    for i, row in df.iterrows():
        original_query = row["query"]

        # **在 query 之后追加 CoT**
        # **在 query 之后追加 CoT**
        cot_prompt = (
            f"{original_query}\n\n"
            "Let's solve the final question step by step as below.\n"
            "Step 1: Identify the relevant financial figures from the provided data.\n"
            "- What are the key numbers needed to answer this question?\n"
            "- Why are these numbers relevant in this financial context?\n"
            "Step 2: Determine the appropriate formula.\n"
            "- What formula should be used to solve this problem?\n"
            "- How does this formula relate to the data selected in Step 1?\n"
            "Step 3: Compute the result.\n"
            "- Apply the formula to the selected data.\n"
            "- Provide the final answer.\n"
        )


        try:
            # **发送 API 请求**
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": cot_prompt}]
            )

            # **解析完整的 Response**
            predicted_response = response.choices[0].message.content.strip()
            predictions.append(predicted_response)

        except Exception as e:
            print(f"❌ 预测失败，错误: {e}")
            predictions.append("error")  # **失败时填充 "error"**

    # **将预测结果添加到 DataFrame**
    df["prediction"] = predictions

    # **保存预测结果 CSV**
    results_file = os.path.join(results_dir, f"{dataset_name}_cot2.csv")  # ✅ 另存为 CoT 版本
    df.to_csv(results_file, index=False)
    print(f"✅ 预测结果已保存至: {results_file}")
