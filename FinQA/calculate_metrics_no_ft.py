import pandas as pd
import re
import os
from sklearn.metrics import accuracy_score

# **文件路径**
results_dir = "results/finqa/base_4o-mini"  # ✅ 替换为不同实验路径，如 `answer_first_test`
results_file = os.path.join(results_dir, "qa_base_4o-mini_predictions.csv")

# **读取测试数据**
df = pd.read_csv(results_file)

# **提取数值（支持负数 & 百分比）**
def extract_numbers(text):
    """ 从文本中提取所有数字（支持负数、小数、百分比）"""
    if pd.isna(text):
        return []
    numbers = re.findall(r'-?\d*\.\d+|-?\d+', text)  # 提取所有数值
    return [float(num) for num in numbers] if numbers else []

# **标准化答案**
def normalize_answer(answer):
    """ 归一化答案格式（去除额外符号，转小写）"""
    return str(answer).lower().strip()

# **判断是否正确**
def is_correct(pred, true):
    """ 判断预测是否正确：
    - 直接数值匹配
    - 允许小误差（百分比误差 <1%）
    - 允许 `Yes/No` 直接匹配 """
    
    # 归一化文本
    normalized_true = normalize_answer(true)
    normalized_pred = normalize_answer(pred)

    # Yes/No 直接匹配
    if normalized_true in ["yes", "no"]:
        return normalized_true in normalized_pred

    # 提取数值
    true_nums = extract_numbers(normalized_true)
    pred_nums = extract_numbers(normalized_pred)

    if true_nums:
        true_num = true_nums[0]  # 仅使用首个正确数值

        for pred_num in pred_nums:
            # 处理百分比（如果预测值含 %，转换为小数）
            if "%" in normalized_pred:
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
        # 无数值时，直接匹配字符串
        return normalized_true in normalized_pred

# **计算 Accuracy**
df["is_correct"] = df.apply(lambda row: is_correct(row["prediction"], row["answer"]), axis=1)
accuracy = df["is_correct"].mean()

# **保存 Accuracy 结果**
metrics_file = os.path.join(results_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")

print(f"✅ Accuracy 计算完成: {accuracy:.4f}")
