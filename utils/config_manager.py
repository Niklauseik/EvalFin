import yaml
import os

class ConfigManager:
    """
    用于加载和管理 config.yaml 配置的类
    """
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"{self.config_path} 文件未找到！")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_api_key(self, service_name):
        """
        从 config.yaml 中获取指定 service 的 api_key
        """
        if service_name not in self.config:
            raise ValueError(f"{service_name} 未在配置文件中找到！")
        return self.config[service_name].get("api_key", None)

    def get_endpoint(self, service_name):
        """
        从 config.yaml 中获取指定 service 的 endpoint
        """
        if service_name not in self.config:
            raise ValueError(f"{service_name} 未在配置文件中找到！")
        return self.config[service_name].get("endpoint", None)

    def get_location(self, service_name):
        """
        从 config.yaml 中获取指定 service 的 location
        """
        if service_name not in self.config:
            raise ValueError(f"{service_name} 未在配置文件中找到！")
        return self.config[service_name].get("location", None)

# 示例用法
if __name__ == "__main__":
    config = ConfigManager()

    openai_key = config.get_api_key("deepseek")
    print(f"OpenAI API Key: {openai_key}")

