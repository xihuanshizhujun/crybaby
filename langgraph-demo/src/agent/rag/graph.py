"""LangGraph图定义，实现检索-反思-迭代工作流"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agent.rag.state import GraphRAGState
from agent.rag.nodes import decompose_query, retrieve, aggregate_results, rerank, web_search, reflect, refine_query, generate_answer
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


def should_use_web_search(state: GraphRAGState) -> str:
    """判断是否需要网络搜索
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    use_web_search = state.get('use_web_search', False)
    retrieved_chunks = state.get('retrieved_chunks', [])
    
    # 如果没有检索到结果，或者结果太少，使用网络搜索
    if use_web_search or len(retrieved_chunks) == 0:
        return "web_search"
    else:
        return "rerank"


def should_continue_retrieve(state: GraphRAGState) -> str:
    """判断是否需要继续检索子查询还是进入后续流程
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    sub_queries = state.get('sub_queries', [])
    sub_query_results = state.get('sub_query_results', {})
    retrieved_chunks = state.get('retrieved_chunks', [])
    
    # 如果有子查询且还未全部检索完，继续检索
    if sub_queries and len(sub_query_results) < len(sub_queries):
        return "retrieve"
    # 如果有子查询且已全部检索完，进入聚合
    elif sub_queries and len(sub_query_results) >= len(sub_queries):
        return "aggregate_results"
    # 没有子查询但有检索结果，判断重排序或网络搜索
    elif not sub_queries and retrieved_chunks:
        use_web_search = state.get('use_web_search', False)
        if use_web_search or len(retrieved_chunks) == 0:
            return "web_search"
        else:
            return "rerank"
    # 没有结果，需要网络搜索
    else:
        return "web_search"


def rerank_or_web_search(state: GraphRAGState) -> str:
    """判断是重排序还是网络搜索
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    use_web_search = state.get('use_web_search', False)
    retrieved_chunks = state.get('retrieved_chunks', [])
    
    if use_web_search or len(retrieved_chunks) == 0:
        return "web_search"
    else:
        return "rerank"


def create_rag_graph() -> StateGraph:
    """创建增强的图RAG工作流图（支持查询分解）
    
    工作流：
    1. START -> decompose_query (查询分解)
    2. decompose_query -> retrieve (检索子查询)
    3. retrieve -> retrieve (继续检索下一个子查询) 或 aggregate_results (所有子查询检索完成)
    4. aggregate_results -> rerank_or_web_search (判断重排序或网络搜索)
    5. rerank_or_web_search -> rerank (重排序) 或 web_search (网络搜索)
    6. rerank -> reflect (反思)
    7. web_search -> reflect (反思)
    8. reflect -> refine_query (如果需要迭代) 或 generate_answer (如果满足)
    9. refine_query -> decompose_query (重新分解查询)
    10. generate_answer -> END
    
    Returns:
        LangGraph StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(GraphRAGState)
    
    # 添加节点
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("aggregate_results", aggregate_results)
    workflow.add_node("rerank", rerank)
    workflow.add_node("web_search", web_search)
    workflow.add_node("reflect", reflect)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("generate_answer", generate_answer)
    
    # 定义边：从查询分解开始
    workflow.set_entry_point("decompose_query")
    workflow.add_edge("decompose_query", "retrieve")
    
    # 条件边：检索后判断是继续检索子查询还是聚合结果
    workflow.add_conditional_edges(
        "retrieve",
        should_continue_retrieve,
        {
            "retrieve": "retrieve",  # 继续检索下一个子查询
            "aggregate_results": "aggregate_results",  # 所有子查询检索完成，聚合结果
            "rerank": "rerank",  # 没有子查询，有结果，重排序
            "web_search": "web_search",  # 没有结果，直接网络搜索
        }
    )
    
    # 聚合结果后判断是重排序还是网络搜索
    workflow.add_conditional_edges(
        "aggregate_results",
        rerank_or_web_search,
        {
            "rerank": "rerank",
            "web_search": "web_search",
        }
    )
    
    # 重排序后进入反思
    workflow.add_edge("rerank", "reflect")
    
    # 网络搜索后进入反思
    workflow.add_edge("web_search", "reflect")
    
    # 条件边：反思后决定是迭代还是生成答案
    workflow.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {
            "refine_query": "refine_query",
            "generate_answer": "generate_answer",
        }
    )
    
    # 迭代路径：优化查询后重新分解
    workflow.add_edge("refine_query", "decompose_query")
    
    # 生成答案后结束
    workflow.add_edge("generate_answer", END)
    
    # 编译图
    graph = workflow.compile()
    
    logger.info("增强的图RAG工作流创建完成（包含查询分解、语义重排序和网络搜索）")
    return graph


# 全局图实例
_rag_graph = None


def get_rag_graph() -> StateGraph:
    """获取图RAG实例（单例）"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = create_rag_graph()
    return _rag_graph


