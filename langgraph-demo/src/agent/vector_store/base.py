"""向量数据库抽象基类，定义统一的接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class VectorStoreError(Exception):
    """向量存储异常"""
    pass


@dataclass
class DocumentChunk:
    """文档分块"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """检索结果"""
    chunk: DocumentChunk
    score: float


class VectorStore(ABC):
    """向量数据库抽象基类
    
    实现开闭原则：对扩展开放，对修改关闭
    新增向量数据库只需实现此接口，无需修改其他代码
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化向量数据库
        
        Returns:
            bool: 是否成功初始化
        """
        pass
    
    @abstractmethod
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """添加文档分块到向量数据库
        
        Args:
            chunks: 文档分块列表
            collection_name: 集合名称，如果为None则使用默认集合
            
        Returns:
            List[str]: 插入的文档ID列表
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """向量相似度搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回top k个结果
            filter_dict: 过滤条件
            collection_name: 集合名称
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """删除文档
        
        Args:
            ids: 文档ID列表
            collection_name: 集合名称
            
        Returns:
            bool: 是否成功删除
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool: 数据库是否健康
        """
        pass
    
    @abstractmethod
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict[str, Any]: 集合信息
        """
        pass


