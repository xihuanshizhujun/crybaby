"""文档分块器，针对金融文档优化，保留金融术语完整性"""

import logging
import re
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import config

logger = logging.getLogger(__name__)


class FinancialChunker:
    """金融文档分块器
    
    特点：
    1. 保留金融术语的完整性
    2. 智能分块，避免截断关键信息
    3. 支持表格特殊处理
    """
    
    # 金融术语模式（用于检测和保留）
    FINANCIAL_PATTERNS = [
        r'财务报表[^。]*。',
        r'[资产负债|利润|现金流].*表[^。]*。',
        r'[营业|净|毛].*[收入|利润|利率][^。]*。',
        r'[投|融]资[^。]*。',
        r'估值[^。]*。',
        r'[A-Z]轮融资[^。]*。',
        r'IPO[^。]*。',
        r'[并|重]购[^。]*。',
    ]
    
    def __init__(self):
        """初始化分块器"""
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
    
    def _preserve_financial_terms(self, text: str) -> str:
        """保留金融术语完整性
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not config.data_processing.preserve_financial_terms:
            return text
        
        # 在金融术语前后添加特殊标记，防止被截断
        processed_text = text
        for pattern in self.FINANCIAL_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 在匹配项前后添加保护标记
                start, end = match.span()
                processed_text = (
                    processed_text[:start] +
                    f"<FINANCIAL_TERM>{match.group()}</FINANCIAL_TERM>" +
                    processed_text[end:]
                )
        
        return processed_text
    
    def _split_text_smart(self, text: str) -> List[str]:
        """智能分块（改进版，避免乱码和截断）
        
        Args:
            text: 原始文本
            
        Returns:
            分块后的文本列表
        """
        if not text or not text.strip():
            return []
        
        # 清理文本：移除控制字符和异常字符
        cleaned_text = self._clean_text(text)
        
        # 保留金融术语
        processed_text = self._preserve_financial_terms(cleaned_text)
        
        # 使用LangChain的分块器
        chunks = self.text_splitter.split_text(processed_text)
        
        # 清理保护标记并处理分块
        cleaned_chunks = []
        for chunk in chunks:
            # 移除保护标记
            cleaned_chunk = chunk.replace("<FINANCIAL_TERM>", "").replace("</FINANCIAL_TERM>", "")
            
            # 再次清理
            cleaned_chunk = self._clean_text(cleaned_chunk)
            
            # 过滤太短或太长的块
            chunk_len = len(cleaned_chunk.strip())
            if chunk_len > 0:
                if chunk_len <= config.data_processing.max_chunk_size:
                    # 如果块太短，记录警告但不丢弃（除非完全为空）
                    if chunk_len < config.data_processing.min_chunk_size:
                        logger.warning(f"块长度 {chunk_len} 小于最小长度 {config.data_processing.min_chunk_size}，但仍保留")
                    cleaned_chunks.append(cleaned_chunk.strip())
        
        return cleaned_chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除乱码和控制字符
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        # 移除控制字符（除了换行符、制表符等常用字符）
        # 保留：\n, \r, \t, 以及所有可打印字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 移除异常的Unicode字符（如零宽字符）
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
        
        # 规范化空白字符
        text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符合并为一个空格
        text = re.sub(r'\n{3,}', '\n\n', text)  # 多个换行符合并为两个
        
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def chunk_document(
        self,
        content: Dict[str, Any],
        doc_id: str,
        user_id: str,
        doc_type: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """分块文档
        
        Args:
            content: 解析后的文档内容
            doc_id: 文档ID
            user_id: 用户ID
            doc_type: 文档类型
            
        Returns:
            分块列表，每个分块包含content和metadata
        """
        chunks = []
        
        # 处理文本内容
        if content.get("text"):
            text_chunks = self._split_text_smart(content["text"])
            
            for idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "user_id": user_id,
                        "doc_type": doc_type,
                        "chunk_index": idx,
                        "chunk_type": "text",
                        "file_type": content.get("file_type", "unknown"),
                    },
                })
        
        # 处理表格（单独作为分块）
        if content.get("tables"):
            text_chunks_count = len([c for c in chunks if c.get("metadata", {}).get("chunk_type") == "text"])
            for idx, table in enumerate(content["tables"]):
                # 将表格转换为文本表示
                if isinstance(table.get("data"), list):
                    table_text = self._table_to_text(table["data"])
                    if table_text:
                        chunks.append({
                            "content": table_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "user_id": user_id,
                                "doc_type": doc_type,
                                "chunk_index": text_chunks_count + idx,
                                "chunk_type": "table",
                                "file_type": content.get("file_type", "unknown"),
                                "table_columns": table.get("columns", []),
                            },
                        })
        
        logger.info(f"文档 {doc_id} 被分为 {len(chunks)} 个块")
        return chunks
    
    def _table_to_text(self, table_data: List[Dict[str, Any]]) -> str:
        """将表格数据转换为文本
        
        Args:
            table_data: 表格数据
            
        Returns:
            文本表示
        """
        if not table_data:
            return ""
        
        text_lines = []
        if isinstance(table_data[0], dict):
            # DataFrame格式
            for row in table_data:
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                text_lines.append(row_text)
        else:
            # 二维列表格式
            for row in table_data:
                if isinstance(row, list):
                    text_lines.append(" | ".join([str(cell) for cell in row]))
        
        return "\n".join(text_lines)


# 全局分块器实例
_chunker = None


def get_chunker() -> FinancialChunker:
    """获取分块器实例（单例）"""
    global _chunker
    if _chunker is None:
        _chunker = FinancialChunker()
    return _chunker


def chunk_documents(
    content: Dict[str, Any],
    doc_id: str,
    user_id: str,
    doc_type: str = "unknown"
) -> List[Dict[str, Any]]:
    """分块文档的便捷函数"""
    chunker = get_chunker()
    return chunker.chunk_document(content, doc_id, user_id, doc_type)


