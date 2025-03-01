import openai
import json

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 替换为你的微调模型 ID
fine_tuned_model = "ft:gpt-4o-2024-08-06:personal:balanced:B3kEd6za"

# **System Prompt**
system_message = "You are a financial sentiment classifier."

# **Few-Shot Examples**
examples = [
    {"role": "user", "content": 'Text: Facebook $FB received a Buy rating from Wells Fargo 5-star Analyst Peter Stabler (Wells Fargo)\nAnswer:'},
    {"role": "assistant", "content": "Positive"},
    
    {"role": "user", "content": 'Text: Short more $FAZ for all who don\'t know that means markets will tank now :-p\nAnswer:'},
    {"role": "assistant", "content": "Negative"},
    
    {"role": "user", "content": 'Text: Numbers looked good, not great, but good. I think conference call will instill confidence\nAnswer:'},
    {"role": "assistant", "content": "Neutral"}
]

# **测试文本**
test_text = "Text: $HAL If bulls lucky enuff to get an upside gap fill, better take it. Wouldn't chase it here.\nAnswer:"

# **组装消息**
messages = [
    {"role": "system", "content": system_message},
    *examples,  # 添加示例
    {"role": "user", "content": test_text}
]

# **打印发送的消息**
print("📤 Sending message to API:")
print(json.dumps(messages, indent=4))

try:
    # **发送 API 请求**
    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=messages
    )

    # **解析模型预测的情感类别**
    predicted_label = response.choices[0].message.content.strip()

    # **打印 API 响应**
    print("\n📥 Model Response:")
    print(predicted_label)

except Exception as e:
    print(f"\n❌ 预测失败，错误: {e}")
