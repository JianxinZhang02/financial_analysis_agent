from utils.config_handler import prompts_cof
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_cof["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e
    

def load_rag_summary_prompts():
    try:
        rag_summary_prompt_path = get_abs_path(prompts_cof["rag_summary_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_summary_prompts]在yaml配置项中没有rag_summary_prompt_path配置项")
        raise e

    try:
        return open(rag_summary_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_summary_prompts]解析RAG摘要提示词出错，{str(e)}")
        raise e

def load_report_prompts():  # 报告提示词加载
    try:
        report_prompt_path = get_abs_path(prompts_cof["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告提示词出错，{str(e)}")
        raise e
    
if __name__ == '__main__':
    print(load_system_prompts())