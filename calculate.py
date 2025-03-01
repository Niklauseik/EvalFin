import pandas as pd
import re
import os

# **文件路径**
results_dir = "results/cqa"
datasets = {
    "with_answer": os.path.join(results_dir, "with_answer", "with_answer.csv"),
    "without_answer": os.path.join(results_dir, "without_answer", "without_answer.csv"),
    "only_question": os.path.join(results_dir, "only_question", "only_question.csv"),
    "without_answer_cot": os.path.join(results_dir, "without_answer_cot", "without_answer_cot.csv"),
    "without_answer_cot2": os.path.join(results_dir, "without_answer_cot2", "without_answer_cot2.csv"),
    "with_answer_cot": os.path.join(results_dir, "with_answer_cot", "with_answer_cot.csv"),
    
}

# **提取数值（去掉 `,` 以支持千位分隔符）**
def extract_numbers(text):
    """ 从文本中提取所有数字（支持负数、小数、百分比），并去掉 `,` 确保格式一致 """
    if pd.isna(text):
        return []
    numbers = re.findall(r'-?\d{1,3}(?:,\d{3})*\.?\d*|-?\d+\.?\d*', text)  # 允许千位分隔符
    numbers = [float(num.replace(',', '')) for num in numbers]  # 去除 `,` 转换为 float
    return numbers if numbers else []

# **判断预测是否正确**
def is_correct(pred, true):
    """ 判断预测是否正确：
    - 直接数值匹配（忽略 `,`）
    - 允许小误差（百分比误差 <1%）
    - 允许 `x` 和 `-x` 作为正确答案（对称数值匹配） """
    
    # 提取数值
    true_nums = extract_numbers(true)
    pred_nums = extract_numbers(pred)

    if true_nums:
        true_num = true_nums[0]  # 仅使用首个正确数值

        for pred_num in pred_nums:
            # 处理百分比（如果预测值含 %，转换为小数）
            if "%" in pred:
                pred_num /= 100
            
            # 允许 1% 误差范围
            tolerance = 0.01 * max(abs(pred_num), abs(true_num))
            if abs(pred_num - true_num) <= tolerance:
                return True

            # 允许 `x` 和 `-x` 作为正确答案（对称数值匹配）
            if abs(abs(pred_num) - abs(true_num)) <= tolerance:
                return True

        return False
    else:
        return False  # 没有数值时，默认错误

# **计算并保存 Accuracy 结果**
for dataset_name, file_path in datasets.items():
    if not os.path.exists(file_path):
        print(f"❌ {dataset_name} 结果文件不存在，跳过计算。")
        continue

    # **读取测试数据**
    df = pd.read_csv(file_path)

    # **计算 Accuracy**
    df["correct"] = df.apply(lambda row: 1 if is_correct(str(row["prediction"]), str(row["answer"])) else 0, axis=1)
    accuracy = df["correct"].mean()

    # **创建存储 `metrics.txt` 的目录**
    dataset_results_dir = os.path.dirname(file_path)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # **保存 Accuracy 结果**
    metrics_file = os.path.join(dataset_results_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")

    # **保存带有 `correct` 标记的完整数据**
    df.to_csv(file_path, index=False)

    print(f"✅ {dataset_name} Accuracy 计算完成: {accuracy:.4f}，已保存至 {metrics_file}")