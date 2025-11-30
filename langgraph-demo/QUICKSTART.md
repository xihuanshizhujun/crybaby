# 快速启动指南

## 📋 配置清单

### 1. 创建 `.env` 文件

从 `env.example` 复制并配置：

```bash
# 在项目根目录下
cp env.example .env
```

**必须配置的关键项：**
- `OPENAI_API_KEY`: 你的 OpenAI API 密钥
- `VECTOR_DB_TYPE`: 选择 `qdrant`、`milvus` 或 `weaviate`
- 对应的向量数据库连接信息

### 2. 启动向量数据库（选择一个）

#### 选项 A: Weaviate（推荐，简单）

```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -v weaviate_data:/var/lib/weaviate \
  semitechnologies/weaviate:latest
```

然后在 `.env` 中配置：
```env
VECTOR_DB_TYPE=weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_COLLECTION=FinancialDoc
```

#### 选项 B: Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

然后在 `.env` 中配置：
```env
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=financial_docs
```

#### 选项 C: Milvus

```bash
# 下载 docker-compose.yml
curl -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed/docker-compose.yml

# 启动
docker-compose up -d
```

然后在 `.env` 中配置：
```env
VECTOR_DB_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=financial_docs
```

### 3. 创建必要的目录

```bash
# 创建上传目录
mkdir -p uploads
```

### 4. 验证 langgraph.json 配置

`langgraph.json` 已经正确配置，指向：
- 图路径: `./src/agent/graph.py:graph` ✅
- 环境文件: `.env` ✅

### 5. 安装依赖（如果还没安装）

```bash
uv add . --dev
```

## 🚀 启动项目

### 方式 1: Streamlit 前端（推荐）

```bash
streamlit run streamlit_app.py
```

访问: http://localhost:8501

**使用流程：**
1. 在侧边栏上传 PDF/DOCX/PPT 文件
2. 等待文件处理完成
3. 在主界面输入问题进行对话

### 方式 2: LangGraph Server（用于调试）

```bash
langgraph dev
```

访问 LangGraph Studio: http://localhost:8123

## 📝 完整配置示例

### `.env` 文件最小配置

```env
# === 必填项 ===
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# === 向量数据库（选择一个）===
VECTOR_DB_TYPE=weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_COLLECTION=FinancialDoc

# === 可选配置 ===
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
RAG_TOP_K=5
```

## ✅ 验证安装

### 1. 检查向量数据库连接

```python
python -c "
from agent.vector_store.factory import VectorStoreFactory
from agent.config import config

store = VectorStoreFactory.create_vector_store()
if store.initialize():
    print('✅ 向量数据库连接成功')
    print(f'数据库类型: {config.vector_db.db_type}')
else:
    print('❌ 连接失败')
"
```

### 2. 检查包导入

```python
python -c "
from agent.config import config
from agent.rag.graph import get_rag_graph
from agent.vector_store.factory import VectorStoreFactory
print('✅ 所有模块导入成功')
"
```

### 3. 测试图结构

```python
python -c "
from agent.graph import graph
print('✅ LangGraph 图加载成功')
print(f'图节点数: {len(graph.nodes)}')
"
```

## 🐛 常见问题

### 问题 1: 导入错误 `ModuleNotFoundError: No module named 'agent'`

**解决：** 确保已安装包
```bash
uv add . --dev
# 或
pip install -e .
```

### 问题 2: 向量数据库连接失败

**解决：** 
1. 检查 docker 容器是否运行：`docker ps`
2. 检查 `.env` 中的配置是否正确
3. 检查端口是否被占用

### 问题 3: LangGraph dev 找不到图

**解决：** 检查 `langgraph.json` 中的路径是否正确指向 `./src/agent/graph.py:graph`

### 问题 4: OpenAI API 错误

**解决：**
1. 检查 `OPENAI_API_KEY` 是否正确
2. 检查 `OPENAI_BASE_URL` 是否可访问
3. 检查 API 余额

## 📦 Docker 命令速查

### Weaviate

```bash
# 启动
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:latest

# 停止
docker stop weaviate

# 删除
docker rm weaviate

# 查看日志
docker logs weaviate

# 进入容器
docker exec -it weaviate sh
```

### Qdrant

```bash
# 启动
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 停止
docker stop qdrant

# Web UI 访问
# http://localhost:6333/dashboard
```

## 🎯 下一步

1. ✅ 配置 `.env` 文件
2. ✅ 启动向量数据库（Docker）
3. ✅ 创建 `uploads` 目录
4. ✅ 运行 `uv add . --dev`
5. ✅ 启动 Streamlit：`streamlit run streamlit_app.py`
6. ✅ 上传文档并开始对话！

---

**提示：** 如果遇到问题，检查日志输出，大部分错误信息会指明问题所在。


