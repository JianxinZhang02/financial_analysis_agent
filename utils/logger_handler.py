import logging
import os
from utils.path_tool import get_abs_path
from datetime import datetime

# 日志存的目录

LOG_ROOT = get_abs_path('logs')


os.makedirs(LOG_ROOT,exist_ok=True)

# 日志的模板配置

DEFAULT_LOGGING_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(name: str='agent', log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    获取日志记录器
    :param name: 记录器名称
    :param log_file: 日志文件路径，如果为None则不输出到文件
    :param level: 日志级别，默认为INFO
    :return: 配置好的日志记录器对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 创建控制台处理器并设置格式   终端handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(console_handler)

    # 如果指定了日志文件路径，则创建文件处理器并设置格式   文件handler
    if not log_file:        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)

    logger.addHandler(file_handler)

    return logger

# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")