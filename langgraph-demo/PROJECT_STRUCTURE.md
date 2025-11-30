# 项目结构说明

## 完整目录结构

```
langgraph-demo/
├── src/
│   └── agent/
│       ├── __init__.py              # 主模块导出
│       ├── graph.py                 # 主图定义（LangGraph入口）
│       ├── config.py                # 配置管理（支持多环境）
│       │
│       ├── data_processor/          # 数据处理模块
│       │   ├── __init__.py
│       │   ├── parser.py            # 文档解析（PDF/DOCX/PPT）
│       │   └── chunker.py           # 文档分块（金融术语优化）
│       │
│       ├── vector_store/            # 向量数据库模块（高可用、高扩展）
│       │   ├── __init__.py
│       │   ├── base.py              # 抽象基类
│       │   ├── factory.py           # 工厂类（支持多数据库）
│       │   ├── qdrant_store.py      # Qdrant实现
│       │   ├── milvus_store.py      # Milvus实现
│       │   ├── weaviate_store.py    # Weaviate实现
│       │   └── ha_store.py          # 高可用包装器
│       │
│       ├── rag/                     # 图RAG核心模块
│       │   ├── __init__.py
│       │   ├── state.py             # 状态定义
│       │   ├── nodes.py             # 节点实现（检索/反思/优化/生成）
│       │   └── graph.py             # 图定义（检索-反思-迭代流程）
│       │
│       └── utils/                   # 工具函数
│           ├── __init__.py
│           └── embedding.py         # 嵌入向量生成
│
├── streamlit_app.py                 # Streamlit前端（极简版）
├── pyproject.toml                   # 项目配置和依赖
├── env.example                      # 环境变量示例
├── langgraph.json                   # LangGraph配置
└── README.md                        # 项目说明

```

## 核心模块说明

### 1. 配置模块 (`config.py`)
- 支持多环境配置
- 支持多种向量数据库配置
- 支持高可用配置（备份主机、复制）

### 2. 向量数据库模块 (`vector_store/`)
- **抽象基类** (`base.py`): 定义统一接口
- **工厂模式** (`factory.py`): 支持动态创建和切换数据库
- **多数据库支持**: Qdrant、Milvus、Weaviate
- **高可用** (`ha_store.py`): 故障转移和备份

**设计原则**:
- 开闭原则：新增数据库只需实现接口
- 单一职责：每个数据库类只负责一种实现
- 依赖倒置：依赖抽象而非具体实现

### 3. 数据处理模块 (`data_processor/`)
- **文档解析** (`parser.py`): 
  - PDF: PyPDF2
  - DOCX: python-docx
  - PPT: python-pptx
  - 表格提取和转换
- **文档分块** (`chunker.py`):
  - 基于LangChain的RecursiveCharacterTextSplitter
  - 金融术语保留（防止截断）
  - 表格特殊处理

### 4. 图RAG核心模块 (`rag/`)
- **状态定义** (`state.py`): TypedDict定义状态流转
- **节点实现** (`nodes.py`):
  - `retrieve`: 向量检索
  - `reflect`: 反思检索结果
  - `refine_query`: 优化查询
  - `generate_answer`: 生成答案
- **图定义** (`graph.py`): 检索-反思-迭代工作流

**工作流程**:
```
用户查询
  ↓
检索相关文档
  ↓
反思：评估检索结果
  ↓
是否需要迭代？
  ├─ 是 → 优化查询 → 重新检索
  └─ 否 → 生成最终答案
```

### 5. Streamlit前端 (`streamlit_app.py`)
- 文件上传（侧边栏）
- 对话界面（主区域）
- 极简设计（<200行代码）

## 技术特性

### 高可用
- 支持主从数据库配置
- 自动故障转移
- 数据复制

### 高扩展
- 抽象接口设计
- 工厂模式
- 插件化架构

### 开闭原则
- 新增向量数据库：只需实现`VectorStore`接口
- 新增文档格式：扩展`DocumentParser`
- 自定义分块策略：修改`FinancialChunker`

## 使用流程

1. **配置环境变量**: 复制`env.example`为`.env`并配置
2. **安装依赖**: `uv add .` 或 `pip install -e .`
3. **启动向量数据库**: 根据需要启动Qdrant/Milvus/Weaviate
4. **运行前端**: `streamlit run streamlit_app.py`
5. **上传文档**: 在侧边栏上传PDF/DOCX/PPT文件
6. **开始对话**: 在主界面提问

## 依赖管理

使用 `uv` 进行依赖管理：
```bash
uv add .              # 安装所有依赖
uv add package-name   # 添加新依赖
```

或使用 `pip`:
```bash
pip install -e .
```

## 配置说明

主要配置项（通过环境变量）:
- `VECTOR_DB_TYPE`: 向量数据库类型
- `OPENAI_API_KEY`: OpenAI API密钥
- `RAG_TOP_K`: 检索top k个结果
- `RAG_MAX_ITERATIONS`: 最大迭代次数

详细配置见 `env.example`


