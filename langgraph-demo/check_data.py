"""检查数据库中的数据"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from agent.config import config
from agent.vector_store.factory import VectorStoreFactory

load_dotenv()

def check_data():
    """检查数据库中的数据"""
    vector_store = VectorStoreFactory.create_vector_store()
    vector_store.initialize()
    
    # 获取集合信息
    info = vector_store.get_collection_info()
    print(f"Collection info: {info}")
    
    # 尝试一个简单的检索
    from agent.utils.embedding import generate_embeddings
    
    test_query = "测试"
    query_embedding = generate_embeddings([test_query])[0]
    
    print(f"\nTesting search with query: '{test_query}'")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"Expected dimension: {config.embedding.dimensions}")
    
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=10,
        filter_dict=None,
    )
    
    print(f"\nSearch results (without threshold filter): {len(results)}")
    for i, result in enumerate(results[:5]):
        print(f"  [{i+1}] Score: {result.score:.4f}, Content: {result.chunk.content[:100]}...")
    
    # 测试带阈值过滤
    threshold = config.rag.similarity_threshold
    print(f"\nWith similarity threshold {threshold}:")
    filtered = [r for r in results if r.score >= threshold]
    print(f"  Filtered results: {len(filtered)}")

if __name__ == "__main__":
    check_data()

