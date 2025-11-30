"""Milvus向量数据库实现"""

import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from agent.vector_store.base import VectorStore, VectorStoreError, DocumentChunk, SearchResult
from agent.config import VectorDBConfig

logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStore):
    """Milvus向量数据库实现"""
    
    def __init__(self, config: VectorDBConfig):
        """初始化Milvus客户端
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.collection_name = config.collection_name
        self._connection_alias = f"milvus_{config.host}_{config.port}"
        self.collection: Optional[Collection] = None
        self._connect()
    
    def _connect(self):
        """连接Milvus"""
        try:
            connections.connect(
                alias=self._connection_alias,
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout,
            )
            logger.info(f"成功连接到Milvus: {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise VectorStoreError(f"无法连接到Milvus: {e}")
    
    def initialize(self) -> bool:
        """初始化集合"""
        try:
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="金融文档向量存储集合"
            )
            
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(
                    name=self.collection_name,
                    using=self._connection_alias
                )
                logger.info(f"集合已存在: {self.collection_name}")
            else:
                # 创建集合
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self._connection_alias
                )
                logger.info(f"创建集合: {self.collection_name}")
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
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
        collection_name = collection_name or self.collection_name
        
        if not self.collection:
            self.initialize()
        
        try:
            ids = []
            contents = []
            embeddings = []
            user_ids = []
            doc_types = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                
                ids.append(chunk.id)
                contents.append(chunk.content)
                embeddings.append(chunk.embedding)
                user_ids.append(chunk.metadata.get("user_id", ""))
                doc_types.append(chunk.metadata.get("doc_type", ""))
            
            if not ids:
                return []
            
            data = [ids, contents, embeddings, user_ids, doc_types]
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"成功添加 {len(ids)} 个文档")
            return ids
            
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
        if not self.collection:
            self.initialize()
        
        try:
            # 构建过滤表达式
            expr = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    if key in ["user_id", "doc_type"]:
                        conditions.append(f"{key} == '{value}'")
                if conditions:
                    expr = " && ".join(conditions)
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "user_id", "doc_type"],
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    chunk = DocumentChunk(
                        id=str(hit.id),
                        content=hit.entity.get("content", ""),
                        metadata={
                            "user_id": hit.entity.get("user_id", ""),
                            "doc_type": hit.entity.get("doc_type", ""),
                        },
                    )
                    search_results.append(
                        SearchResult(chunk=chunk, score=hit.score)
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
        if not self.collection:
            self.initialize()
        
        try:
            expr = f'id in {ids}'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"成功删除 {len(ids)} 个文档")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise VectorStoreError(f"删除失败: {e}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            utility.list_collections()
            return True
        except Exception:
            return False
    
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合信息"""
        collection_name = collection_name or self.collection_name
        
        try:
            if not utility.has_collection(collection_name):
                return {"name": collection_name, "error": "集合不存在"}
            
            collection = Collection(name=collection_name, using=self._connection_alias)
            collection.load()
            
            stats = collection.num_entities
            return {
                "name": collection_name,
                "vectors_count": stats,
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"name": collection_name, "error": str(e)}


