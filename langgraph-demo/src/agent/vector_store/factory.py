"""向量数据库工厂类，支持高可用和多数据库切换"""

from typing import Optional, List
from agent.vector_store.base import VectorStore
from agent.vector_store.qdrant_store import QdrantVectorStore
from agent.vector_store.milvus_store import MilvusVectorStore
from agent.vector_store.weaviate_store import WeaviateVectorStore
from agent.config import config, VectorDBConfig


class VectorStoreFactory:
    """向量数据库工厂类
    
    实现工厂模式，支持：
    1. 多数据库类型（Qdrant、Milvus、Weaviate）
    2. 高可用（故障转移）
    3. 动态切换数据库
    """
    
    @staticmethod
    def create_vector_store(
        db_config: Optional[VectorDBConfig] = None
    ) -> VectorStore:
        """创建向量数据库实例
        
        Args:
            db_config: 数据库配置，如果为None则使用全局配置
            
        Returns:
            VectorStore: 向量数据库实例
        """
        if db_config is None:
            db_config = config.vector_db
        
        if db_config.db_type == "qdrant":
            return QdrantVectorStore(db_config)
        elif db_config.db_type == "milvus":
            return MilvusVectorStore(db_config)
        elif db_config.db_type == "weaviate":
            return WeaviateVectorStore(db_config)
        else:
            raise ValueError(f"不支持的数据库类型: {db_config.db_type}")
    
    @staticmethod
    def create_high_availability_store(
        primary_config: VectorDBConfig,
        backup_configs: Optional[List[VectorDBConfig]] = None
    ) -> VectorStore:
        """创建高可用向量数据库实例（带备份）
        
        Args:
            primary_config: 主数据库配置
            backup_configs: 备份数据库配置列表
            
        Returns:
            VectorStore: 高可用向量数据库实例
        """
        # 优先使用主数据库
        primary_store = VectorStoreFactory.create_vector_store(primary_config)
        
        # 如果启用高可用且有备份，使用高可用包装器
        if backup_configs and primary_config.enable_replication:
            from agent.vector_store.ha_store import HighAvailabilityVectorStore
            backup_stores = [
                VectorStoreFactory.create_vector_store(backup)
                for backup in backup_configs
            ]
            return HighAvailabilityVectorStore(primary_store, backup_stores)
        
        return primary_store

