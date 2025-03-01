import pandas as pd
import os

# **文件路径**
results_dir = "results/cqa"
file_with = os.path.join(results_dir, "with_answer", "with_answer.csv")
file_without = os.path.join(results_dir, "without_answer", "without_answer.csv")
study_file = os.path.join(results_dir, "study_samples.csv")

# **加载数据**
df_with = pd.read_csv(file_with)
df_without = pd.read_csv(file_without)

# **确保数据行数匹配**
if len(df_with) != len(df_without):
    raise ValueError("❌ `with_answer` 和 `without_answer` 数据集行数不匹配，无法逐行比较！")

# **筛选 `with_answer` 正确但 `without_answer` 错误的样本**
df_study = df_with[(df_with["correct"] == 1) & (df_without["correct"] == 0)]

# **合并关键信息**
df_study = df_study[["query", "answer", "prediction"]].rename(columns={"prediction": "prediction_with_answer"})
df_study["prediction_without_answer"] = df_without.loc[df_study.index, "prediction"]

# **保存筛选出的样本**
df_study.to_csv(study_file, index=False)

print(f"✅ 已筛选出 `with_answer` 正确但 `without_answer` 失败的样本，共 {len(df_study)} 条，存储至: {study_file}")
