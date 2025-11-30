"""高可用向量存储包装器，支持故障转移和复制"""

import logging
from typing import List, Dict, Any, Optional
from agent.vector_store.base import VectorStore, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


class HighAvailabilityVectorStore(VectorStore):
    """高可用向量存储包装器
    
    实现故障转移和读写分离
    """
    
    def __init__(self, primary: VectorStore, backups: List[VectorStore]):
        """初始化高可用存储
        
        Args:
            primary: 主数据库
            backups: 备份数据库列表
        """
        self.primary = primary
        self.backups = backups
        self.current_store = primary
    
    def _get_available_store(self) -> VectorStore:
        """获取可用的数据库（故障转移）"""
        if self.current_store.health_check():
            return self.current_store
        
        # 尝试切换到备份
        for backup in self.backups:
            if backup.health_check():
                logger.warning(f"主数据库不可用，切换到备份: {backup}")
                self.current_store = backup
                return backup
        
        # 如果主数据库和备份都不可用，尝试恢复主数据库
        if self.primary.health_check():
            logger.info("主数据库已恢复")
            self.current_store = self.primary
            return self.primary
        
        raise Exception("所有向量数据库都不可用")
    
    def initialize(self) -> bool:
        """初始化所有数据库"""
        results = []
        try:
            results.append(self.primary.initialize())
        except Exception as e:
            logger.error(f"初始化主数据库失败: {e}")
            results.append(False)
        
        for backup in self.backups:
            try:
                results.append(backup.initialize())
            except Exception as e:
                logger.error(f"初始化备份数据库失败: {e}")
                results.append(False)
        
        return any(results)
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """添加文档（写入主数据库和备份）"""
        store = self._get_available_store()
        result = store.add_documents(chunks, collection_name)
        
        # 异步复制到备份（简化实现，实际应该异步）
        for backup in self.backups:
            try:
                if backup.health_check():
                    backup.add_documents(chunks, collection_name)
            except Exception as e:
                logger.warning(f"复制到备份失败: {e}")
        
        return result
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """搜索（从当前可用数据库读取）"""
        store = self._get_available_store()
        return store.search(query_embedding, top_k, filter_dict, collection_name)
    
    def delete(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """删除文档"""
        store = self._get_available_store()
        result = store.delete(ids, collection_name)
        
        # 同步删除备份
        for backup in self.backups:
            try:
                if backup.health_check():
                    backup.delete(ids, collection_name)
            except Exception as e:
                logger.warning(f"从备份删除失败: {e}")
        
        return result
    
    def health_check(self) -> bool:
        """健康检查"""
        return self.primary.health_check() or any(b.health_check() for b in self.backups)
    
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合信息"""
        store = self._get_available_store()
        return store.get_collection_info(collection_name)


