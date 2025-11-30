"""Qdrant向量数据库实现"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from agent.vector_store.base import VectorStore, VectorStoreError, DocumentChunk, SearchResult
from agent.config import VectorDBConfig

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant向量数据库实现"""
    
    def __init__(self, config: VectorDBConfig):
        """初始化Qdrant客户端
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.collection_name
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Qdrant客户端"""
        try:
            self.client = QdrantClient(
                url=f"http://{self.config.host}:{self.config.port}",
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        except Exception as e:
            logger.error(f"初始化Qdrant客户端失败: {e}")
            raise VectorStoreError(f"无法连接到Qdrant: {e}")
    
    def initialize(self) -> bool:
        """初始化集合"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # text-embedding-3-small维度
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"创建集合: {self.collection_name}")
            else:
                logger.info(f"集合已存在: {self.collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise VectorStoreError(f"初始化失败: {e}")
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """添加文档"""
        collection = collection_name or self.collection_name
        
        try:
            points = [
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        **chunk.metadata,
                        "content": chunk.content,
                    }
                )
                for chunk in chunks
                if chunk.embedding is not None
            ]
            
            if not points:
                return []
            
            self.client.upsert(
                collection_name=collection,
                points=points,
            )
            
            logger.info(f"成功添加 {len(points)} 个文档到 {collection}")
            return [chunk.id for chunk in chunks if chunk.embedding is not None]
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise VectorStoreError(f"添加文档失败: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """向量搜索"""
        collection = collection_name or self.collection_name
        
        try:
            # 构建过滤条件
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions) if conditions else None
            
            results = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
            )
            
            search_results = []
            for result in results:
                payload = result.payload
                chunk = DocumentChunk(
                    id=str(result.id),
                    content=payload.get("content", ""),
                    metadata={k: v for k, v in payload.items() if k != "content"},
                    embedding=None,
                )
                search_results.append(
                    SearchResult(chunk=chunk, score=result.score)
                )
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise VectorStoreError(f"搜索失败: {e}")
    
    def delete(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """删除文档"""
        collection = collection_name or self.collection_name
        
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=ids,
            )
            logger.info(f"成功删除 {len(ids)} 个文档")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise VectorStoreError(f"删除失败: {e}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合信息"""
        collection = collection_name or self.collection_name
        
        try:
            collection_info = self.client.get_collection(collection)
            return {
                "name": collection,
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"name": collection, "error": str(e)}


