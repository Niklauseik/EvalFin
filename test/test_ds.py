from openai import OpenAI

client = OpenAI(
    api_key="sk-a30a03a5670a4e46a8b2f8decf130411",  # 直接填入你的 OpenAI API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="deepseek-r1",
    messages=[{"role": "user", "content": "9.9和9.11谁大"}]
)

print("思考过程：")
print(completion.choices[0].message.reasoning_content)
