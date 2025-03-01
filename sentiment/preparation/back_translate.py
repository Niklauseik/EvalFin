"""
back_translate.py

放置于: EVALFIN/sentiment/back_translate.py
依赖:
  - config_manager.py (位于 EVALFIN/utils/config_manager.py)
  - config.yaml       (位于 EVALFIN/config.yaml)
  - negative_400.csv  (位于 EVALFIN/datasets/negative_400.csv)
"""

import os
import sys
import time
import uuid
import requests
import pandas as pd
from retry import retry

# 将 EVALFIN/utils 加入搜索路径，便于导入 config_manager
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "utils")
    )
)
from config_manager import ConfigManager  # 您已实现的配置管理器


class AzureTranslator:
    """
    使用 Microsoft Translator Text API（Azure Cognitive Services）的翻译工具类
    """
    def __init__(self, config_manager, service_name='translator'):
        # 从 config_manager 获取配置信息
        self.subscription_key = config_manager.get_api_key(service_name)
        self.endpoint = config_manager.get_endpoint(service_name)
        self.location = config_manager.get_location(service_name)

        if not self.subscription_key or not self.endpoint or not self.location:
            raise ValueError("Azure Translator 配置信息不完整，请检查 config.yaml！")

    @retry(tries=3, delay=2, backoff=2, exceptions=(Exception,))
    def translate_text(self, text, from_lang, to_lang):
        """
        调用 Azure Translator Text API，将文本从 from_lang 翻译成 to_lang
        """
        path = '/translate?api-version=3.0'
        params = f"&from={from_lang}&to={to_lang}"
        constructed_url = self.endpoint + path + params

        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-Type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        body = [{'text': text}]

        response = requests.post(constructed_url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()

        return result[0]['translations'][0]['text']

    def back_translate_en_zh_en(self, text):
        """
        回译函数：英 -> 中 -> 英
        """
        try:
            # 第一步：英 -> 中
            cn_text = self.translate_text(text, 'en', 'zh-Hans')
            time.sleep(1)  # 简单防护，避免速率限制
            # 第二步：中 -> 英
            en_text = self.translate_text(cn_text, 'zh-Hans', 'en')
            print(f"[回译] {text} -> {cn_text} -> {en_text}")
            return en_text
        except Exception as e:
            print(f"[翻译错误] {e}")
            return text  # 出错时返回原文


def main():
    # 1. 创建配置管理器（读取 config.yaml）
    config_manager = ConfigManager('config.yaml')

    # 2. 创建 Azure 翻译器
    translator = AzureTranslator(config_manager, service_name='translator')

    # 3. 读取要处理的 CSV（示例: negative_400.csv）
    input_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "datasets", "negative_400.csv")
    )
    df = pd.read_csv(input_csv)

    # 假设原文列名为 'text_only'
    if 'text_only' not in df.columns:
        raise ValueError("输入 CSV 中找不到 'text_only' 列，请检查列名！")

    # 4. 回译并存储到新列
    df['back_translated_text'] = df['text_only'].apply(translator.back_translate_en_zh_en)

    # 5. 保存结果
    output_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "datasets", "negative_400_backtranslated.csv")
    )
    df.to_csv(output_csv, index=False)
    print(f"回译完成，已保存至: {output_csv}")


if __name__ == '__main__':
    main()
