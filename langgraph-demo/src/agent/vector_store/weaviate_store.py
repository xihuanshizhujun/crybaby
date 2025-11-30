"""Weaviate向量数据库实现"""

import logging
import uuid as uuid_lib
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from agent.vector_store.base import VectorStore, VectorStoreError, DocumentChunk, SearchResult
from agent.config import VectorDBConfig, config

logger = logging.getLogger(__name__)


class WeaviateVectorStore(VectorStore):
    """Weaviate向量数据库实现"""
    
    def __init__(self, config: VectorDBConfig):
        """初始化Weaviate客户端
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.client: Optional[weaviate.WeaviateClient] = None
        self.class_name = config.collection_name
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Weaviate客户端"""
        try:
            # 优先尝试本地连接
            if not self.config.api_key:
                # 本地连接（无认证）
                self.client = weaviate.connect_to_local(
                    host=self.config.host,
                    port=self.config.port,
                )
                logger.info(f"成功连接到Weaviate本地: {self.config.host}:{self.config.port}")
            else:
                # 尝试自定义连接
                try:
                    self.client = weaviate.connect_to_custom(
                        http_host=self.config.host,
                        http_port=self.config.port,
                        http_secure=False,
                        grpc_host=self.config.host,
                        grpc_port=self.config.port + 1000,
                        auth_credentials=weaviate.AuthApiKey(api_key=self.config.api_key),
                    )
                    logger.info(f"成功连接到Weaviate自定义: {self.config.host}:{self.config.port}")
                except Exception:
                    # 如果自定义连接失败，尝试云连接
                    connection_url = f"https://{self.config.host}"
                    if "weaviate.cloud" in self.config.host or self.config.port == 443:
                        self.client = weaviate.connect_to_wcs(
                            cluster_url=connection_url,
                            auth_credentials=weaviate.AuthApiKey(api_key=self.config.api_key),
                        )
                        logger.info(f"成功连接到Weaviate云: {connection_url}")
                    else:
                        raise
        except Exception as e:
            logger.error(f"连接Weaviate失败: {e}")
            raise VectorStoreError(f"无法连接到Weaviate: {e}")
    
    def initialize(self) -> bool:
        """初始化集合（Class）"""
        try:
            if not self.client:
                self._initialize_client()
            
            # 检查类是否存在
            if self.client.collections.exists(self.class_name):
                logger.info(f"类已存在: {self.class_name}")
                return True
            
            # 创建类
            self.client.collections.create(
                name=self.class_name,
                description="金融文档向量存储类",
                vectorizer_config=Configure.Vectorizer.none(),  # 使用外部向量
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )
            
            logger.info(f"创建类: {self.class_name}")
            return True
            
        except Exception as e:
            logger.error(f"初始化类失败: {e}")
            raise VectorStoreError(f"初始化失败: {e}")
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """添加文档"""
        class_name = collection_name or self.class_name
        
        try:
            if not self.client:
                self._initialize_client()
            
            collection = self.client.collections.get(class_name)
            
            # 准备数据对象
            objects_to_insert = []
            inserted_ids = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"跳过没有向量的文档块: {chunk.id}")
                    continue
                
                # 验证向量维度（使用配置中的维度）
                expected_dimensions = config.embedding.dimensions
                if len(chunk.embedding) != expected_dimensions:
                    logger.warning(f"文档块 {chunk.id} 的向量维度不正确: {len(chunk.embedding)}, 期望: {expected_dimensions}")
                    continue
                
                # 处理 UUID：确保是有效的 UUID 格式
                try:
                    # 尝试将 chunk.id 转换为 UUID
                    if isinstance(chunk.id, str):
                        # 如果已经是 UUID 格式，直接使用
                        try:
                            chunk_uuid = uuid_lib.UUID(chunk.id)
                        except ValueError:
                            # 如果不是有效的 UUID，生成一个新的
                            chunk_uuid = uuid_lib.uuid4()
                            logger.debug(f"文档块 ID {chunk.id} 不是有效 UUID，已生成新 UUID: {chunk_uuid}")
                    else:
                        chunk_uuid = uuid_lib.uuid4()
                except Exception as e:
                    logger.warning(f"处理文档块 ID 失败: {e}，生成新 UUID")
                    chunk_uuid = uuid_lib.uuid4()
                
                obj_properties = {
                    "content": chunk.content,
                    "user_id": chunk.metadata.get("user_id", ""),
                    "doc_type": chunk.metadata.get("doc_type", ""),
                    "doc_id": chunk.metadata.get("doc_id", chunk.id),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                }
                
                # 添加其他元数据
                for key, value in chunk.metadata.items():
                    if key not in ["user_id", "doc_type", "doc_id", "chunk_index"]:
                        obj_properties[key] = str(value)
                
                objects_to_insert.append(
                    weaviate.classes.data.DataObject(
                        properties=obj_properties,
                        vector=chunk.embedding,
                        uuid=chunk_uuid,
                    )
                )
                inserted_ids.append(str(chunk_uuid))
            
            if not objects_to_insert:
                expected_dimensions = config.embedding.dimensions
                logger.warning(f"没有可插入的文档对象。输入块数: {len(chunks)}, "
                             f"有效块数: {len([c for c in chunks if c.embedding is not None and len(c.embedding) == expected_dimensions])}")
                return []
            
            # 批量插入（分批处理，避免一次性插入太多）
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(objects_to_insert), batch_size):
                batch = objects_to_insert[i:i + batch_size]
                try:
                    collection.data.insert_many(batch)
                    total_inserted += len(batch)
                    logger.debug(f"成功插入批次 {i // batch_size + 1}: {len(batch)} 个文档")
                except Exception as batch_error:
                    logger.error(f"批次 {i // batch_size + 1} 插入失败: {batch_error}")
                    # 尝试逐个插入，找出问题文档
                    for obj in batch:
                        try:
                            collection.data.insert(obj)
                            total_inserted += 1
                        except Exception as single_error:
                            logger.error(f"单个文档插入失败: {single_error}, UUID: {obj.uuid}")
            
            logger.info(f"成功添加 {total_inserted}/{len(objects_to_insert)} 个文档到 {class_name}")
            return inserted_ids[:total_inserted]
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}", exc_info=True)
            raise VectorStoreError(f"添加文档失败: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """向量搜索"""
        class_name = collection_name or self.class_name
        
        try:
            if not self.client:
                self._initialize_client()
            
            collection = self.client.collections.get(class_name)
            
            # 构建过滤条件（使用 Weaviate v4+ 的 Filter API）
            from weaviate.classes.query import Filter
            
            where_filter = None
            if filter_dict:
                filter_conditions = []
                for key, value in filter_dict.items():
                    if key in ["user_id", "doc_type"]:
                        # 使用 Weaviate v4+ 的 Filter API
                        filter_conditions.append(
                            Filter.by_property(key).equal(str(value))
                        )
                
                if len(filter_conditions) == 1:
                    where_filter = filter_conditions[0]
                elif len(filter_conditions) > 1:
                    # 多个条件用 AND 组合
                    where_filter = Filter.all_of(filter_conditions)
            
            # 执行向量搜索（Weaviate v4+ API）
            # 注意：在 Weaviate v4 中，near_vector() 不接受 where 参数
            # 解决方案：先执行向量搜索获取更多结果，然后在 Python 层面过滤
            
            # 计算需要获取的候选数量（如果有过滤条件，需要更多候选）
            candidate_limit = top_k * 5 if where_filter else top_k
            
            # 执行向量搜索
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=candidate_limit,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["content", "user_id", "doc_type", "doc_id", "chunk_index"],
            )
            
            search_results = []
            for obj in response.objects:
                properties = obj.properties
                
                # 如果有过滤条件，在 Python 层面进行过滤
                if where_filter and filter_dict:
                    # 检查是否匹配过滤条件
                    match = True
                    for key, value in filter_dict.items():
                        if key in ["user_id", "doc_type"]:
                            if properties.get(key) != str(value):
                                match = False
                                break
                    if not match:
                        continue  # 跳过不匹配的结果
                
                metadata = {
                    "user_id": properties.get("user_id", ""),
                    "doc_type": properties.get("doc_type", ""),
                    "doc_id": properties.get("doc_id", ""),
                    "chunk_index": properties.get("chunk_index", 0),
                }
                
                chunk = DocumentChunk(
                    id=str(obj.uuid),
                    content=properties.get("content", ""),
                    metadata=metadata,
                )
                
                # 计算相似度分数（Weaviate返回的是距离，需要转换为相似度）
                distance = obj.metadata.distance if obj.metadata.distance else 1.0
                score = 1.0 - distance  # 将距离转换为相似度
                
                search_results.append(SearchResult(chunk=chunk, score=score))
            
            # 如果有过滤条件，只返回前 top_k 个结果
            if where_filter:
                search_results = search_results[:top_k]
            
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
        class_name = collection_name or self.class_name
        
        try:
            if not self.client:
                self._initialize_client()
            
            collection = self.client.collections.get(class_name)
            
            # 批量删除
            for doc_id in ids:
                try:
                    collection.data.delete_by_id(doc_id)
                except Exception as e:
                    logger.warning(f"删除文档 {doc_id} 失败: {e}")
            
            logger.info(f"成功删除 {len(ids)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise VectorStoreError(f"删除失败: {e}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.client:
                self._initialize_client()
            
            # Weaviate的健康检查
            is_ready = self.client.is_ready()
            return is_ready
            
        except Exception:
            return False
    
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合信息"""
        class_name = collection_name or self.class_name
        
        try:
            if not self.client:
                self._initialize_client()
            
            if not self.client.collections.exists(class_name):
                return {"name": class_name, "error": "类不存在"}
            
            collection = self.client.collections.get(class_name)
            
            # 获取对象数量
            count_response = collection.aggregate.over_all(total_count=True)
            total_count = count_response.total_count if hasattr(count_response, 'total_count') else 0
            
            return {
                "name": class_name,
                "vectors_count": total_count,
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"name": class_name, "error": str(e)}
    
    def __del__(self):
        """关闭连接"""
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass

