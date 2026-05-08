import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex (file_path: str) -> str:
    """
    计算文件的MD5值
    :param file_path: 文件路径
    :return: 文件的MD5值
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None

    if not os.path.isfile(file_path):
        logger.error(f"路径不是文件: {file_path}")
        return None
    

    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:    # 二进制读取
            # 逐块读取文件内容并更新MD5哈希对象
            for byte_block in iter(lambda: f.read(4096), b""):  # 4KB分片，避免文件过大
                md5_hash.update(byte_block)
            return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"计算文件{file_path}的MD5时发生错误: {e}")
        return None 

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):        # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
