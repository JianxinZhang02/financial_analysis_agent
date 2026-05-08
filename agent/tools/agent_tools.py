import random
import os

from langchain_core.tools import tool
from rag.rag_service import RagSummaryService
from utils.config_handler import agent_cof
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

rag = RagSummaryService()

userid = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
months = ["2025-1", "2025-2", "2025-3", "2025-4", "2025-5", "2025-6", "2025-7", "2025-8", "2025-9", "2025-10", "2025-11", "2025-12"]
external_data = {}

@tool(description = "根据用户的查询，使用RAG技术从知识库中检索相关文档，并生成摘要返回给用户")
def rag_summarize(query: str) -> str:
    # Implementation for the RAG summarize tool
    return rag.rag_summarize(query)


@tool(description = "根据用户提供的城市名称，返回该城市的当前天气情况") 
def get_weather(city: str) -> str:
    # Implementation for the weather tool
    return f"The current weather in {city} is sunny with a high of 25°C and a low of 15°C."

@tool(description="返回用户的当前位置")
def get_location() ->str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州"])


@tool(description="获取用户ID，以字符串形式返回")
def get_user_id() -> str:
    return random.choice(userid)

@tool(description="获取当前月份，以字符串形式返回")
def get_current_month() -> str:

    return random.choice(months)

def generate_external_data() -> str:
    """
    组成为一个dict，
    {
        'userid': {
            'month1': {'特征': '值1', '效率': '值2', ...}
            'month2': {'特征': '值1', '效率': '值2', ...}
            'month3': {'特征': '值1', '效率': '值2', ...}
        },
        'userid': {
            'month1': {'特征': '值1', '效率': '值2', ...}
            'month2': {'特征': '值1', '效率': '值2', ...}
            'month3': {'特征': '值1', '效率': '值2', ...}
        },
        ...
    }
    """

    if not external_data:
        external_path = get_abs_path(agent_cof['external_data_path'])
        if not os.path.exists(external_path):
            raise FileNotFoundError(f"外部数据文件 {external_path} 不存在，请检查配置项external_data_path")
        with open(external_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:  # 跳过第一行表头
                line :list[str] = line.strip().split(',')

                userid = line[0].strip().replace('"', '')  # 用户ID
                feature = line[1].strip().replace('"', '')  # 特征
                efficiency = line[2].strip().replace('"', '')  # 效率
                consumables = line[3].strip().replace('"', '')  # 耗材
                comparison = line[4].strip().replace('"', '')  # 对比
                month = line[5].strip().replace('"', '')  # 月份

                if userid not in external_data: # 如果用户ID不在external_data中，先创建一个空的字典
                    external_data[userid] = {}

                external_data[userid][month] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }

@tool(description="生成特定用户的外部报告")
def fetch_external_data(userid: str, month: str) -> str:
    generate_external_data()  # 调用一哈
    try:
        return external_data[userid][month]
    except KeyError:
        logger.warning(f"未找到用户ID {userid} 在月份 {month} 的数据")
        return ''


# if __name__ == "__main__":
#     print(fetch_external_data("1021", "2025-01"))