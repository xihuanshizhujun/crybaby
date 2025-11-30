"""文本分割器 - 基于 deep-searcher 实现"""
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import config

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分割器"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.data_processing.chunk_size,
            chunk_overlap=config.data_processing.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # 段落
                "\n",    # 换行
                "。",    # 中文句号
                "！",    # 中文感叹号
                "？",    # 中文问号
                ". ",    # 英文句号
                "! ",    # 英文感叹号
                "? ",    # 英文问号
                "; ",    # 分号
                ", ",    # 逗号
                " ",     # 空格
                "",      # 字符级
            ],
        )
    
    def split_text(self, text: str) -> List[str]:
        """分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        # 过滤太短的块
        min_chunk_size = config.data_processing.min_chunk_size
        filtered_chunks = [
            chunk.strip() 
            for chunk in chunks 
            if len(chunk.strip()) >= min_chunk_size
        ]
        
        return filtered_chunks
    
    def split_documents(
        self,
        content: Dict[str, Any],
        doc_id: str,
        user_id: str,
        doc_type: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """分割文档
        
        Args:
            content: 文档内容（包含 text 字段）
            doc_id: 文档ID
            user_id: 用户ID
            doc_type: 文档类型
            
        Returns:
            文档块列表
        """
        text = content.get('text', '')
        if not text:
            logger.warning(f"Document {doc_id} has no text content")
            return []
        
        chunks = self.split_text(text)
        
        result = []
        for idx, chunk_text in enumerate(chunks):
            result.append({
                "content": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "doc_type": doc_type,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            })
        
        logger.info(f"Split document {doc_id} into {len(result)} chunks")
        return result


# 全局实例
_text_splitter = None

def get_text_splitter() -> TextSplitter:
    """获取文本分割器单例"""
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = TextSplitter()
    return _text_splitter
