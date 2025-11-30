"""Streamlit前端应用（极简版）"""

import streamlit as st
import os
import uuid
from pathlib import Path

# 导入项目模块
from agent.data_processor.file_manager import FileManager
from agent.data_processor.parser import DocumentParser  # 兼容性导入
from agent.data_processor.chunker import chunk_documents
from agent.vector_store.factory import VectorStoreFactory
from agent.vector_store.base import DocumentChunk
from agent.utils.embedding import generate_embeddings
from agent.rag.graph import get_rag_graph
from agent.rag.state import GraphRAGState
from agent.config import config
from langchain_core.messages import HumanMessage


# 页面配置
st.set_page_config(
    page_title="金融图RAG智能问答",
    page_icon="💼",
    layout="wide"
)

# 初始化session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


def process_uploaded_files(uploaded_files):
    """处理已选择的上传文件，将其写入向量数据库"""
    if not uploaded_files:
        st.warning("请先选择要上传的文档。")
        return

    vector_store = VectorStoreFactory.create_vector_store()
    vector_store.initialize()

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_chunks = []

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"正在处理: {uploaded_file.name}")

        # 保存文件
        upload_dir = Path(config.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # 使用FileManager解析文档（支持更多格式，更好的中文支持）
            if not FileManager.is_supported(str(file_path)):
                st.error(f"不支持的文件格式: {uploaded_file.name}")
                continue
            
            content = FileManager.parse_file(str(file_path))

            # 分块
            doc_id = str(uuid.uuid4())
            chunks = chunk_documents(
                content=content,
                doc_id=doc_id,
                user_id=st.session_state.user_id,
                doc_type=uploaded_file.name.split(".")[-1],
            )

            # 生成嵌入向量
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = generate_embeddings(chunk_texts)

            # 创建DocumentChunk对象
            # 使用UUID作为ID，确保Weaviate兼容性
            document_chunks = []
            for i, chunk in enumerate(chunks):
                # 生成唯一的UUID作为ID
                chunk_uuid = str(uuid.uuid4())
                document_chunks.append(
                    DocumentChunk(
                        id=chunk_uuid,
                        content=chunk["content"],
                        metadata={
                            **chunk["metadata"],
                            "original_chunk_id": f"{doc_id}_{chunk['metadata']['chunk_index']}",  # 保留原始ID在metadata中
                        },
                        embedding=embeddings[i],
                    )
                )

            # 插入向量数据库
            vector_store.add_documents(document_chunks)
            all_chunks.extend(chunks)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        except Exception as e:
            st.error(f"处理文件 {uploaded_file.name} 失败: {e}")

    status_text.text("处理完成！")
    st.success(f"成功处理 {len(uploaded_files)} 个文件，共 {len(all_chunks)} 个文档块")

    # 清理临时文件
    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name
        if file_path.exists():
            file_path.unlink()


def chat_interface():
    """对话界面（上下布局 + 上传/发送按钮）"""
    st.markdown("### 💬 上下文对话")

    # 对话历史区域
    chat_container = st.container()
    with chat_container:
        if not st.session_state.conversation_history:
            st.info("还没有对话内容，先上传文档或直接开始提问吧。")
        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            st.markdown(f"**问：** {question}")
            st.markdown(f"**答：** {answer}")
            st.divider()

    st.markdown("---")

    # 下半部分：输入 + 上传 + 按钮
    with st.container():
        user_input = st.text_area("在这里输入你的问题：", key="user_query", height=100)

        # 上传控件 + 按钮一行
        col_upload, col_send = st.columns([1, 1])

        with col_upload:
            uploaded_files = st.file_uploader(
                "选择要上传的文档（PDF/DOCX/DOC/PPT/Excel/TXT）",
                type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt"],
                accept_multiple_files=True,
                key="chat_uploader",
            )
            upload_clicked = st.button("📁 上传并入库", use_container_width=True)

        with col_send:
            send_clicked = st.button("📨 发送对话", type="primary", use_container_width=True)

    # 处理上传按钮逻辑
    if upload_clicked:
        with st.spinner("正在上传并处理文档..."):
            process_uploaded_files(uploaded_files)

    # 处理发送对话按钮逻辑
    if send_clicked and user_input:
        with st.spinner("正在思考..."):
            try:
                # 调用RAG图
                rag_graph = get_rag_graph()

                initial_state = GraphRAGState(
                    messages=[HumanMessage(content=user_input)],
                    query=user_input,
                    user_id=st.session_state.user_id,
                    retrieved_chunks=[],
                    retrieval_scores=[],
                    reflection_result=None,
                    needs_iteration=False,
                    iteration_count=0,
                    refined_query=None,
                    final_answer=None,
                    metadata={},
                )

                result = rag_graph.invoke(initial_state)
                answer = result.get("final_answer", "抱歉，无法生成答案。")

                # 保存到历史
                st.session_state.conversation_history.append((user_input, answer))

                # 刷新页面以展示新对话
                st.rerun()

            except Exception as e:
                st.error(f"生成答案失败: {e}")


st.markdown("## 💼 金融图RAG智能问答系统")
st.caption(f"会话用户 ID：`{st.session_state.user_id[:8]}...`")

# 主内容区：上下文对话界面
chat_interface()
