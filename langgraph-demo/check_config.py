"""配置检查脚本，验证项目配置是否正确"""

import os
import sys
from pathlib import Path


def check_env_file():
    """检查 .env 文件"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env 文件不存在")
        print("   请运行: cp env.example .env")
        return False
    
    print("✅ .env 文件存在")
    
    # 检查关键配置
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ["OPENAI_API_KEY"]
    missing = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)
    
    if missing:
        print(f"⚠️  缺少关键配置: {', '.join(missing)}")
        return False
    
    print("✅ 关键配置已设置")
    return True


def check_imports():
    """检查模块导入"""
    try:
        from agent.config import config
        from agent.rag.graph import get_rag_graph
        from agent.vector_store.factory import VectorStoreFactory
        print("✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请运行: uv add . --dev")
        return False


def check_langgraph():
    """检查 LangGraph 图"""
    try:
        from agent.graph import graph
        print("✅ LangGraph 图加载成功")
        return True
    except Exception as e:
        print(f"❌ LangGraph 图加载失败: {e}")
        return False


def check_uploads_dir():
    """检查 uploads 目录"""
    uploads_path = Path("uploads")
    if not uploads_path.exists():
        print("⚠️  uploads 目录不存在，正在创建...")
        uploads_path.mkdir(exist_ok=True)
    
    print("✅ uploads 目录存在")
    return True


def check_vector_db():
    """检查向量数据库连接"""
    try:
        from agent.vector_store.factory import VectorStoreFactory
        from agent.config import config
        
        store = VectorStoreFactory.create_vector_store()
        if store.health_check():
            print(f"✅ 向量数据库连接成功 ({config.vector_db.db_type})")
            return True
        else:
            print(f"⚠️  向量数据库连接失败 ({config.vector_db.db_type})")
            print(f"   请检查 {config.vector_db.host}:{config.vector_db.port} 是否运行")
            return False
    except Exception as e:
        print(f"⚠️  向量数据库检查失败: {e}")
        print("   请确保向量数据库已启动（Docker）")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("🔍 检查项目配置...")
    print("=" * 50)
    print()
    
    results = []
    
    # 检查 uploads 目录
    results.append(("上传目录", check_uploads_dir()))
    print()
    
    # 检查 .env 文件
    results.append(("环境配置", check_env_file()))
    print()
    
    # 检查模块导入
    results.append(("模块导入", check_imports()))
    print()
    
    # 检查 LangGraph
    results.append(("LangGraph", check_langgraph()))
    print()
    
    # 检查向量数据库（可选，不阻塞）
    results.append(("向量数据库", check_vector_db()))
    print()
    
    # 总结
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"✅ 所有检查通过 ({passed}/{total})")
        return 0
    else:
        print(f"⚠️  部分检查未通过 ({passed}/{total})")
        print("\n请根据上述提示修复问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())


