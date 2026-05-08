import os
from utils.logger_handler import logger
from langchain_chroma import Chroma
from utils.config_handler import chroma_cof
from model.factory import embed_model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.path_tool import get_abs_path

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma( # 创建Chroma向量数据库实例
            collection_name=chroma_cof['collection_name'],
            embedding_function=embed_model,  # 这里需要根据实际情况传入一个文本嵌入函数
            persist_directory=chroma_cof['persist_directory'] # 初始化向量数据库连接
            )  
        
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_cof['chunk_size'],
            chunk_overlap=chroma_cof['chunk_overlap'],
            separators=chroma_cof['separators'],
            length_function = len
        )  # 初始化文本分割器


    def get_retriever(self):    # 获取检索器，默认返回top3相关文档，可以根据需要调整search_kwargs参数
        return self.vector_store.as_retriever(search_kwargs={'k':chroma_cof['retriever_k']})
    
    def load_doucments(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        
        def check_md5_hex(check_file):    # 传入一个文件路径，计算MD5十六进制字符串，并与MD5存储文件中的记录进行对比，判断是否需要重新处理该文件
            if not os.path.exists(chroma_cof['md5_hex_store']):   # 说明还没有MD5存储文件，第一次处理，直接返回False
                # 创建一个空的MD5存储文件
                open(get_abs_path(chroma_cof['md5_hex_store']), 'w', encoding='utf-8').close()  # 创建一个空的MD5存储文件
                return False
            with open(get_abs_path(chroma_cof['md5_hex_store']), 'r', encoding='utf-8') as f:
                for line in f:
                    record_file = line.strip()
                    if record_file == check_file:
                        return True # md5匹配，说明文件未修改过，无需重新处理
            return False
        
        def save_md5_hex(file_md5_hex):   # 将文件的MD5十六进制字符串保存到MD5存储文件中，每行一个记录
            with open(get_abs_path(chroma_cof['md5_hex_store']), 'a', encoding='utf-8') as f:   # 追加模式
                f.write(file_md5_hex + '\n')

        def get_file_docunments(file):   # 根据文件路径，调用相应的加载器获取文档列表   即 文件->langchain中的Document对象列表
            if file.endswith('.pdf'):
                return pdf_loader(file)
            elif file.endswith('.txt'):
                return txt_loader(file)
            else:
                logger.warning(f"不支持的文件类型{file}，跳过")
                return []
            
        file_list: list = listdir_with_allowed_type(get_abs_path(chroma_cof['data_path']), tuple(chroma_cof['allow_knowledge_file_type']))  # 获取数据文件夹内的所有允许的文件类型的文件列表
        for file in file_list:
            file_md5 = get_file_md5_hex(file)  # 计算文件的MD5十六进制字符串
            if not file_md5:
                continue    # 计算MD5失败，跳过该文件
            if check_md5_hex(file_md5):   # 检查MD5十六进制字符串是否已经存在于MD5存储文件中，存在说明文件未修改过，无需重新处理
                logger.info(f"文件{file}已经存在库中，跳过处理")
                continue
            documents: list[Document] = get_file_docunments(file)   # 获取文件对应的文档列表
            if not documents:
                logger.warning(f"文件{file}没有成功加载出文档，跳过")
                continue
            for doc in documents:
                doc.metadata['source'] = file   # 在文档的metadata中添加一个字段source，记录该文档对应的文件路径
            split_docs = self.spliter.split_documents(documents)    # 对文档列表进行文本分割，得到分割后的文档列表
            
            # 将内容存入向量库中
            self.vector_store.add_documents(split_docs)    # 将分割后的文档列表添加

            save_md5_hex(file_md5)   # 将文件的MD5十六进制字符串保存到MD5存储文件中，避免下次重复处理

            logger.info(f"文件{file}处理完成，已添加{len(split_docs)}条记录到向量库中")



if __name__ == '__main__':
    vector_store_service = VectorStoreService()
    vector_store_service.load_doucments()
    retriver = vector_store_service.get_retriever()
    res = retriver.invoke("迷路")  # 测试检索器，输入一个查询，看看能否返回相关的文档
    for r in res:
        print(r.metadata['source'], r.page_content[:100])
        print('---'*20)