"""LangGraph图定义，实现检索-反思-迭代工作流"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agent.rag.state import GraphRAGState
from agent.rag.nodes import retrieve, reflect, refine_query, generate_answer
from agent.config import config

logger = logging.getLogger(__name__)


def should_continue_reflection(state: GraphRAGState) -> str:
    """判断是否需要继续反思迭代
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    needs_iteration = state.get('needs_iteration', False)
    iteration_count = state.get('iteration_count', 0)
    
    if needs_iteration and iteration_count < config.rag.max_iterations:
        return "refine_query"
    else:
        return "generate_answer"


def create_rag_graph() -> StateGraph:
    """创建图RAG工作流图
    
    工作流：
    1. START -> retrieve (检索)
    2. retrieve -> reflect (反思)
    3. reflect -> refine_query (如果需要迭代) 或 generate_answer (如果满足)
    4. refine_query -> retrieve (重新检索)
    5. generate_answer -> END
    
    Returns:
        LangGraph StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(GraphRAGState)
    
    # 添加节点
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("reflect", reflect)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("generate_answer", generate_answer)
    
    # 定义边
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reflect")
    
    # 条件边：反思后决定是迭代还是生成答案
    workflow.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {
            "refine_query": "refine_query",
            "generate_answer": "generate_answer",
        }
    )
    
    # 迭代路径：优化查询后重新检索
    workflow.add_edge("refine_query", "retrieve")
    
    # 生成答案后结束
    workflow.add_edge("generate_answer", END)
    
    # 编译图
    graph = workflow.compile()
    
    logger.info("图RAG工作流创建完成")
    return graph


# 全局图实例
_rag_graph = None


def get_rag_graph() -> StateGraph:
    """获取图RAG实例（单例）"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = create_rag_graph()
    return _rag_graph


