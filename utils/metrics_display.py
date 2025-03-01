import pandas as pd
import matplotlib.pyplot as plt
import base64

# 1️⃣ 读取 Base64 编码的文件并解码
input_base64_file = "Step Metrics from OpenAI API.csv"  # 修改为你的 base64 编码的 CSV 文件路径
output_decoded_file = "decoded_step_metrics.csv"  # 解码后存储的 CSV 文件名

# 读取 Base64 编码的内容并解码
with open(input_base64_file, "r") as f:
    base64_content = f.read().strip()  # 去掉可能的空格或换行符

# 解码 Base64 并存储为 CSV
decoded_bytes = base64.b64decode(base64_content)
with open(output_decoded_file, "wb") as f:
    f.write(decoded_bytes)

print(f"✅ Base64 解码完成，已保存到: {output_decoded_file}")

# 2️⃣ 读取解码后的 CSV 文件
df = pd.read_csv(output_decoded_file)

# 3️⃣ 检查数据结构
print("📊 Step Metrics CSV 前 5 行数据：")
print(df.head())

# 4️⃣ 处理数据，确保所有数值列是 float 类型
df = df.apply(pd.to_numeric, errors='coerce')

# 5️⃣ 绘制训练损失和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["train_loss"], label="Training Loss", color="green")
if "valid_loss" in df.columns and df["valid_loss"].notna().sum() > 0:
    plt.plot(df["step"], df["valid_loss"], label="Validation Loss", color="purple")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)

# 6️⃣ 显示损失曲线
plt.show()

# 7️⃣ 统计最终训练损失 & 验证损失
final_train_loss = df["train_loss"].iloc[-1]
final_valid_loss = df["valid_loss"].dropna().iloc[-1] if "valid_loss" in df.columns and df["valid_loss"].notna().sum() > 0 else None

print(f"✅ 最终训练损失 (train_loss): {final_train_loss:.4f}")
if final_valid_loss is not None:
    print(f"✅ 最终验证损失 (valid_loss): {final_valid_loss:.4f}")

# 8️⃣ 进一步分析：
# 如果 valid_loss 明显高于 train_loss，可能存在过拟合。
# 如果 train_loss 和 valid_loss 都较高，可能需要调整超参数（学习率、batch size）。
