"""金融图RAG主图定义，整合所有模块"""

from __future__ import annotations

from typing import Any, Dict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from agent.rag.graph import get_rag_graph
from agent.rag.state import GraphRAGState

# 导出主要的图
def create_main_graph():
    """创建主图（直接使用RAG图）"""
    return get_rag_graph()


# LangGraph标准接口：graph变量
graph = create_main_graph()


def invoke_rag(query: str, user_id: str = None) -> Dict[str, Any]:
    """便捷函数：调用RAG系统
    
    Args:
        query: 用户查询
        user_id: 用户ID（可选）
        
    Returns:
        RAG结果
    """
    rag_graph = get_rag_graph()
    
    initial_state = GraphRAGState(
        messages=[HumanMessage(content=query)],
        query=query,
        user_id=user_id,
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
    return result
