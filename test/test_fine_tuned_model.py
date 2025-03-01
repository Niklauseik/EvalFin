import openai
from utils import config_manager


config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 替换为你的微调模型 ID
fine_tuned_model = "ft:gpt-3.5-turbo-0125:personal:balanced:B3hYdOlm"

# 发送请求
response = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[{"role": "user", "content": "What is the sentiment of the following financial post: Positive, Negative, or Neutral?\nText: Whats up with $LULU?  Numbers looked good, not great, but good.  I think conference call will instill confidence.\nAnswer:"}]
)

# 输出结果
print("Model Prediction:", response.choices[0].message.content)
