"""
yaml 形如  k:v
"""

import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path('config/rag.yaml'), encoding: str='utf-8') -> dict:
    """
    加载RAG智能体的配置文件
    :param config_path: 配置文件的相对路径
    :param encoding: 文件编码
    :return: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) # 返回键值对好像是

def load_chroma_config(config_path: str=get_abs_path('config/chroma.yaml'), encoding: str='utf-8') -> dict:
    """
    加载Chroma数据库的配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) # 返回键值对好像是


def load_prompts_config(config_path: str=get_abs_path('config/prompts.yaml'), encoding: str='utf-8') -> dict:
    """
    加载提示词的配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) # 返回键值对好像是


def load_agent_config(config_path: str=get_abs_path('config/agent.yaml'), encoding: str='utf-8') -> dict:
    """
    加载Agent的配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) # 返回键值对好像是
    
rag_cof = load_rag_config()
chroma_cof = load_chroma_config()
prompts_cof = load_prompts_config()
agent_cof = load_agent_config()

if __name__ == '__main__':
    print("RAG配置：", rag_cof['chat_model_name'])