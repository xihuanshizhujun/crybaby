"""测试 Weaviate 连接和上传功能"""

import sys
import logging
from agent.config import config
from agent.vector_store.factory import VectorStoreFactory

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_weaviate_connection():
    """测试 Weaviate 连接"""
    print("=" * 50)
    print("测试 Weaviate 连接")
    print("=" * 50)
    
    print(f"配置信息:")
    print(f"  - db_type: {config.vector_db.db_type}")
    print(f"  - host: {config.vector_db.host}")
    print(f"  - port: {config.vector_db.port}")
    print(f"  - collection: {config.vector_db.collection_name}")
    print(f"  - api_key: {'已设置' if config.vector_db.api_key else '未设置'}")
    print()
    
    try:
        # 创建向量存储实例
        print("1. 创建向量存储实例...")
        vector_store = VectorStoreFactory.create_vector_store()
        print("   ✓ 创建成功")
        print()
        
        # 初始化（创建 collection）
        print("2. 初始化集合...")
        result = vector_store.initialize()
        print(f"   ✓ 初始化成功: {result}")
        print()
        
        # 健康检查
        print("3. 健康检查...")
        health = vector_store.health_check()
        print(f"   ✓ 健康状态: {health}")
        print()
        
        # 获取集合信息
        print("4. 获取集合信息...")
        info = vector_store.get_collection_info()
        print(f"   ✓ 集合信息: {info}")
        print()
        
        # 测试插入一个简单的文档
        print("5. 测试插入文档...")
        from agent.vector_store.base import DocumentChunk
        import uuid
        
        # 创建一个测试向量（1536维，全0）
        test_embedding = [0.0] * 1536
        
        test_chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            content="这是一个测试文档",
            metadata={
                "user_id": "test_user",
                "doc_type": "test",
                "doc_id": "test_doc_1",
                "chunk_index": 0,
            },
            embedding=test_embedding,
        )
        
        inserted_ids = vector_store.add_documents([test_chunk])
        print(f"   ✓ 插入成功，ID: {inserted_ids}")
        print()
        
        print("=" * 50)
        print("所有测试通过！")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"   ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 50)
        print("测试失败")
        print("=" * 50)
        return False

if __name__ == "__main__":
    success = test_weaviate_connection()
    sys.exit(0 if success else 1)





