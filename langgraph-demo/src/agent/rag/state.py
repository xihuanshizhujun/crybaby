"""图RAG状态定义"""

from typing import TypedDict, List, Dict, Any, Optional,Annotated
from langchain_core.messages import BaseMessage
from operator import add


class GraphRAGState(TypedDict):
    """图RAG状态
    
    状态流转：
    1. 用户提问 -> query
    2. 检索 -> retrieved_chunks
    3. 反思 -> reflection_result, needs_iteration
    4. 迭代 -> refined_query (如果需要)
    5. 生成答案 -> final_answer
    """
    # 输入
    messages: List[BaseMessage]
    query: str
    user_id: Optional[str]
    
    # 查询分解
    sub_queries: List[str]  # 子查询列表
    sub_query_results: Dict[str, List[Dict[str, Any]]]  # 每个子查询的检索结果
    
    # 检索结果
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    
    # 重排序结果（语义增强）
    reranked_chunks: Annotated[List[Dict[str, Any]],add]
    
    # 网络搜索结果
    web_search_results: List[Dict[str, Any]]
    use_web_search: bool
    
    # 反思阶段
    reflection_result: Optional[str]
    needs_iteration: bool
    
    # 迭代阶段
    iteration_count: int
    refined_query: Optional[str]
    
    # 最终答案
    final_answer: Optional[str]
    
    # 元数据
    metadata: Dict[str, Any]


