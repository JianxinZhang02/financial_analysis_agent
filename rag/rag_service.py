"""
总结服务类：用户提问，搜索参考资料，将**提问和参考资料**提交给模型，让模型总结回复
"""

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_summary_prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from model.factory import chat_model


def print_promt(prompt):
    print("-"*50)
    print(prompt)
    print("-"*50)
    return prompt



class RagSummaryService:
    def __init__(self):
        self.vector_store = VectorStoreService()    # 初始化向量库服务
        self.retriever = self.vector_store.get_retriever()    # 初始化检索器

        self.prompt_text = load_rag_summary_prompts()    # 初始化提示词文本
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)    # 初始化提示词模板

        self.model = chat_model    # 初始化语言模型
        self.chain = self.__init__chain()    # 初始化链式调用对象

    
    def __init__chain(self):
        # 初始化链式调用对象

        chain = self.prompt_template | print_promt | self.model | StrOutputParser()  # 将提示词模板和语言模型连接成一个链式调用对象
        return chain

    # chain 有了 后续提供核心功能：拿到参考资料，让模型去总结

    def retrieve_docs(self, query:str) -> list[Document]:
        # 1. 使用检索器获取相关文档
        relevant_docs = self.retriever.invoke(query)
        return relevant_docs


    def rag_summarize(self, query:str) -> str:
        # 2. 将提问和相关文档输入到链式调用对象中，让模型生成总结回复
        relevant_docs = self.retrieve_docs(query)    # 获取相关文档
        relevant_docs_content = "\n\n".join([doc.page_content for doc in relevant_docs])    # 将相关文档的内容拼接成一个字符串，作为模型输入的一部分

        # 构建模型输入
        model_input = {
            "input": query,
            "context": relevant_docs_content
        }

        # 调用链式调用对象，生成总结回复
        summary = self.chain.invoke(model_input)
        return summary
    

if __name__ == '__main__':
    rag_summary_service = RagSummaryService()
    query = "小户型适合那种扫地机器人呢？"
    summary = rag_summary_service.rag_summarize(query)
    print("用户提问：", query)
    print("RAG总结回复：", summary)