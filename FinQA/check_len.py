import json
import tiktoken

# 选择合适的 tokenizer（GPT-4o & GPT-3.5 使用的 BPE）
tokenizer = tiktoken.get_encoding("cl100k_base")

# 目标 JSONL 文件路径
jsonl_files = [
    "datasets/finqa_jsonl/finqa_answer_first.jsonl",
    "datasets/finqa_jsonl/finqa_cot_first.jsonl"
]

TOKEN_LIMIT = 4096  # OpenAI 微调最大 token 限制
token_counts = {}

for file_path in jsonl_files:
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    tokens_per_sample = []
    for line in lines:
        data = json.loads(line)
        messages = data["messages"]
        
        # 计算每条消息的 token 数
        total_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in messages)
        tokens_per_sample.append(total_tokens)

    # 统计信息
    max_tokens = max(tokens_per_sample)
    min_tokens = min(tokens_per_sample)
    avg_tokens = sum(tokens_per_sample) / len(tokens_per_sample)
    over_limit = sum(1 for x in tokens_per_sample if x > TOKEN_LIMIT)
    
    token_counts[file_path] = {
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "avg_tokens": avg_tokens,
        "total_samples": len(tokens_per_sample),
        "over_limit": over_limit,
        "over_limit_percentage": (over_limit / len(tokens_per_sample)) * 100
    }

# 打印结果
for file, stats in token_counts.items():
    print(f"📂 文件: {file}")
    print(f"   - 最大 token 数: {stats['max_tokens']}")
    print(f"   - 最小 token 数: {stats['min_tokens']}")
    print(f"   - 平均 token 数: {stats['avg_tokens']:.2f}")
    print(f"   - 总样本数: {stats['total_samples']}")
    print(f"   - 超过 {TOKEN_LIMIT} tokens 的样本数: {stats['over_limit']} ({stats['over_limit_percentage']:.2f}%)\n")
