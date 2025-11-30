"""诊断上传流程的详细问题"""

import sys
import logging
import uuid
from pathlib import Path

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_upload_flow():
    """诊断整个上传流程"""
    print("=" * 60)
    print("诊断上传流程")
    print("=" * 60)
    
    try:
        from agent.config import config
        from agent.vector_store.factory import VectorStoreFactory
        from agent.data_processor.parser import DocumentParser
        from agent.data_processor.chunker import chunk_documents
        from agent.utils.embedding import generate_embeddings
        from agent.vector_store.base import DocumentChunk
        
        print("\n1. 检查配置...")
        print(f"   - db_type: {config.vector_db.db_type}")
        print(f"   - host: {config.vector_db.host}")
        print(f"   - port: {config.vector_db.port}")
        print(f"   - collection: {config.vector_db.collection_name}")
        print(f"   - embedding_model: {config.embedding.model_name}")
        print(f"   - embedding_dimensions: {config.embedding.dimensions}")
        print(f"   - chunk_size: {config.data_processing.chunk_size}")
        
        print("\n2. 测试向量数据库连接...")
        vector_store = VectorStoreFactory.create_vector_store()
        init_result = vector_store.initialize()
        print(f"   - 初始化结果: {init_result}")
        
        health = vector_store.health_check()
        print(f"   - 健康检查: {health}")
        
        print("\n3. 模拟文档解析和分块...")
        # 创建一个模拟的文档内容
        test_content = {
            "text": "这是一个测试文档。包含一些金融术语，比如营业收入、净利润、资产负债表等。",
            "tables": [],
            "file_type": "test"
        }
        
        doc_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        doc_type = "test"
        
        print(f"   - doc_id: {doc_id}")
        print(f"   - user_id: {user_id}")
        print(f"   - doc_type: {doc_type}")
        
        chunks = chunk_documents(
            content=test_content,
            doc_id=doc_id,
            user_id=user_id,
            doc_type=doc_type,
        )
        print(f"   - 分块数量: {len(chunks)}")
        
        if len(chunks) == 0:
            print("   ERROR: 分块后没有生成任何块！")
            return False
        
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"   - 块 {i}: 长度={len(chunk['content'])}, metadata={chunk['metadata']}")
        
        print("\n4. 测试向量生成...")
        chunk_texts = [chunk["content"] for chunk in chunks]
        print(f"   - 准备生成 {len(chunk_texts)} 个向量...")
        
        try:
            embeddings = generate_embeddings(chunk_texts)
            print(f"   - 成功生成 {len(embeddings)} 个向量")
            
            if len(embeddings) != len(chunk_texts):
                print(f"   ERROR: 向量数量({len(embeddings)}) != 文本数量({len(chunk_texts)})")
                return False
            
            # 检查向量维度
            for i, emb in enumerate(embeddings):
                if len(emb) != config.embedding.dimensions:
                    print(f"   ERROR: 向量 {i} 维度错误: {len(emb)} != {config.embedding.dimensions}")
                    return False
                if not isinstance(emb, list):
                    print(f"   ERROR: 向量 {i} 不是列表类型: {type(emb)}")
                    return False
                if not all(isinstance(x, (int, float)) for x in emb):
                    print(f"   ERROR: 向量 {i} 包含非数字元素")
                    return False
            
            print(f"   - 所有向量维度正确: {len(embeddings[0])}")
            
        except Exception as e:
            print(f"   ERROR: 生成向量失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n5. 创建 DocumentChunk 对象...")
        document_chunks = []
        for i, chunk in enumerate(chunks):
            # 检查ID格式问题
            chunk_id = f"{doc_id}_{chunk['metadata']['chunk_index']}"
            print(f"   - 块 {i}: ID={chunk_id}, embedding存在={embeddings[i] is not None}")
            
            doc_chunk = DocumentChunk(
                id=chunk_id,
                content=chunk["content"],
                metadata=chunk["metadata"],
                embedding=embeddings[i],
            )
            document_chunks.append(doc_chunk)
        
        print(f"   - 创建了 {len(document_chunks)} 个 DocumentChunk")
        
        print("\n6. 测试上传到向量数据库...")
        try:
            inserted_ids = vector_store.add_documents(document_chunks)
            print(f"   - 成功插入 {len(inserted_ids)} 个文档")
            print(f"   - 插入的ID示例: {inserted_ids[:3] if inserted_ids else 'None'}")
            return True
        except Exception as e:
            print(f"   ERROR: 上传失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"ERROR: 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_upload_flow()
    print("\n" + "=" * 60)
    if success:
        print("诊断完成：所有步骤都成功！")
    else:
        print("诊断完成：发现问题，请查看上面的错误信息")
    print("=" * 60)
    sys.exit(0 if success else 1)





