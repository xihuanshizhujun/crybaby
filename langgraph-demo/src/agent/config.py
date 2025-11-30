"""配置管理模块，支持多环境配置和高可用设置"""

import os
from typing import Literal, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    db_type: Literal["qdrant", "milvus", "weaviate"]
    host: str
    port: int
    collection_name: str
    api_key: Optional[str] = None
    timeout: int = 30
    # 高可用配置
    backup_hosts: Optional[list] = None
    enable_replication: bool = False


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536
    timeout: int = 30


@dataclass
class DataProcessingConfig:
    """数据处理配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 金融术语特殊处理
    preserve_financial_terms: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


@dataclass
class RAGConfig:
    """RAG配置"""
    top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.4  # 降低阈值，使检索更宽松
    max_iterations: int = 3
    enable_reflection: bool = True
    enable_iteration: bool = True


class Config:
    """全局配置类"""
    
    def __init__(self):
        # LLM配置
        self.llm = LLMConfig(
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY1"),
        )
        
        # 嵌入配置
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        )
        
        # 向量数据库配置
        db_type = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()
        if db_type == "qdrant":
            backup_hosts = os.getenv("QDRANT_BACKUP_HOSTS")
            backup_hosts = backup_hosts.split(",") if backup_hosts else None
            self.vector_db = VectorDBConfig(
                db_type="qdrant",
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name=os.getenv("QDRANT_COLLECTION", "financial_docs"),
                api_key=os.getenv("QDRANT_API_KEY"),
                backup_hosts=backup_hosts,
                enable_replication=os.getenv("QDRANT_REPLICATION", "false").lower() == "true",
            )
        elif db_type == "milvus":
            self.vector_db = VectorDBConfig(
                db_type="milvus",
                host=os.getenv("MILVUS_HOST", "localhost"),
                port=int(os.getenv("MILVUS_PORT", "19530")),
                collection_name=os.getenv("MILVUS_COLLECTION", "financial_docs"),
                api_key=os.getenv("MILVUS_API_KEY"),
            )
        else:  # weaviate
            backup_hosts = os.getenv("WEAVIATE_BACKUP_HOSTS")
            backup_hosts = backup_hosts.split(",") if backup_hosts else None
            self.vector_db = VectorDBConfig(
                db_type="weaviate",
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                collection_name=os.getenv("WEAVIATE_COLLECTION", "FinancialDoc"),
                api_key=os.getenv("WEAVIATE_API_KEY"),
                backup_hosts=backup_hosts,
                enable_replication=os.getenv("WEAVIATE_REPLICATION", "false").lower() == "true",
            )
        
        # 数据处理配置
        self.data_processing = DataProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            preserve_financial_terms=True,
        )
        
        # RAG配置
        self.rag = RAGConfig(
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "3")),
            similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.4")),  # 默认0.4，平衡检索严格度和召回率
            max_iterations=int(os.getenv("RAG_MAX_ITERATIONS", "3")),
        )
        
        # 其他配置
        self.upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        self.user_id_field = os.getenv("USER_ID_FIELD", "user_id")


# 全局配置实例
config = Config()

