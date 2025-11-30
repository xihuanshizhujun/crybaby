"""嵌入向量生成工具"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from agent.config import config


def generate_embedding(text: str) -> List[float]:
    """生成单个文本的嵌入向量
    
    Args:
        text: 输入文本
        
    Returns:
        嵌入向量
    """
    embedding_model = OpenAIEmbeddings(
        model=config.embedding.model_name,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.base_url,
    )
    return embedding_model.embed_query(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """批量生成嵌入向量
    
    Args:
        texts: 文本列表
        
    Returns:
        嵌入向量列表
    """
    embedding_model = OpenAIEmbeddings(
        model=config.embedding.model_name,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.base_url,
    )
    return embedding_model.embed_documents(texts)


