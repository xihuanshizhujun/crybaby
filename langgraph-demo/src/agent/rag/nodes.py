"""图RAG节点实现"""

import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.rag.state import GraphRAGState
from agent.vector_store.factory import VectorStoreFactory
from agent.config import config

logger = logging.getLogger(__name__)


# 全局LLM和Embedding实例
_llm = None
_embedding = None
_vector_store = None


def get_llm():
    """获取LLM实例（单例）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
    return _llm


def get_embedding():
    """获取Embedding实例（单例）"""
    global _embedding
    if _embedding is None:
        _embedding = OpenAIEmbeddings(
            model=config.embedding.model_name,
            openai_api_key=config.llm.api_key,
            openai_api_base=config.llm.base_url,
        )
    return _embedding


def get_vector_store():
    """获取向量存储实例（单例）"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreFactory.create_vector_store()
        _vector_store.initialize()
    return _vector_store


def retrieve(state: GraphRAGState) -> Dict[str, Any]:
    """检索节点：从向量数据库检索相关文档
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    logger.info(f"开始检索，查询: {state['query']}")
    
    try:
        # 生成查询向量
        embedding_model = get_embedding()
        query_embedding = embedding_model.embed_query(state['query'])
        
        # 构建过滤条件
        filter_dict = None
        if state.get('user_id'):
            filter_dict = {"user_id": state['user_id']}
        
        # 向量检索
        vector_store = get_vector_store()
        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=config.rag.top_k,
            filter_dict=filter_dict,
        )
        
        # 过滤低相似度结果
        retrieved_chunks = []
        retrieval_scores = []
        
        for result in search_results:
            if result.score >= config.rag.similarity_threshold:
                retrieved_chunks.append({
                    "content": result.chunk.content,
                    "metadata": result.chunk.metadata,
                    "score": result.score,
                })
                retrieval_scores.append(result.score)
        
        logger.info(f"检索到 {len(retrieved_chunks)} 个相关文档块")
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "retrieval_scores": retrieval_scores,
        }
        
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return {
            "retrieved_chunks": [],
            "retrieval_scores": [],
        }


def reflect(state: GraphRAGState) -> Dict[str, Any]:
    """反思节点：评估检索结果是否满足查询需求
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    if not config.rag.enable_reflection:
        return {
            "reflection_result": "反思已禁用",
            "needs_iteration": False,
        }
    
    logger.info("开始反思阶段")
    
    try:
        llm = get_llm()
        
        # 构建反思提示词
        retrieved_texts = "\n\n".join([
            f"[文档 {i+1}]\n{chunk['content'][:500]}\n相似度: {chunk['score']:.3f}"
            for i, chunk in enumerate(state.get('retrieved_chunks', [])[:3])
        ])
        
        reflection_prompt = f"""你是一个金融文档检索质量评估专家。请评估以下检索结果是否充分回答了用户的问题。

用户问题: {state['query']}

检索到的文档:
{retrieved_texts if retrieved_texts else "未检索到相关文档"}

请从以下角度评估：
1. 检索结果的相关性（是否与问题相关）
2. 信息的完整性（是否包含回答问题的关键信息）
3. 是否需要更精确的检索策略

请用简洁的中文回答：
- 如果检索结果充分，回答"检索充分，可以直接生成答案"
- 如果需要改进，指出问题并建议改进方向

评估结果："""
        
        reflection_result = llm.invoke([HumanMessage(content=reflection_prompt)]).content
        
        # 判断是否需要迭代
        needs_iteration = (
            "不充分" in reflection_result or
            "需要改进" in reflection_result or
            "不足" in reflection_result or
            len(state.get('retrieved_chunks', [])) == 0
        ) and state.get('iteration_count', 0) < config.rag.max_iterations
        
        logger.info(f"反思结果: {reflection_result[:100]}...")
        logger.info(f"是否需要迭代: {needs_iteration}")
        
        return {
            "reflection_result": reflection_result,
            "needs_iteration": needs_iteration,
        }
        
    except Exception as e:
        logger.error(f"反思失败: {e}")
        return {
            "reflection_result": f"反思过程出错: {e}",
            "needs_iteration": False,
        }


def refine_query(state: GraphRAGState) -> Dict[str, Any]:
    """优化查询节点：基于反思结果优化查询
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    logger.info("开始优化查询")
    
    try:
        llm = get_llm()
        
        refine_prompt = f"""基于以下反思结果，请优化用户的查询问题，使其更适合文档检索。

原始问题: {state['query']}

反思结果: {state.get('reflection_result', '')}

请生成一个更精确的查询问题，要求：
1. 保持问题的核心意图
2. 增加更具体的关键词
3. 如果原始问题模糊，使其更加明确

优化后的问题："""
        
        refined_query = llm.invoke([HumanMessage(content=refine_prompt)]).content.strip()
        
        # 更新迭代计数
        iteration_count = state.get('iteration_count', 0) + 1
        
        logger.info(f"优化后的查询: {refined_query}")
        
        return {
            "refined_query": refined_query,
            "query": refined_query,  # 更新查询
            "iteration_count": iteration_count,
        }
        
    except Exception as e:
        logger.error(f"优化查询失败: {e}")
        return {
            "refined_query": state['query'],  # 保持原查询
            "iteration_count": state.get('iteration_count', 0) + 1,
        }


def generate_answer(state: GraphRAGState) -> Dict[str, Any]:
    """生成答案节点：基于检索结果生成最终答案
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    logger.info("开始生成答案")
    
    try:
        llm = get_llm()
        
        # 构建上下文
        context_texts = "\n\n".join([
            f"[来源文档 {i+1}]\n{chunk['content']}\n"
            for i, chunk in enumerate(state.get('retrieved_chunks', []))
        ])
        
        # 构建系统提示词（金融领域特化）
        system_prompt = """你是一个专业的金融投资顾问AI助手，擅长分析企业财报、投融资报告等金融文档。

你的职责：
1. 基于提供的文档内容，准确回答用户问题
2. 回答要专业、准确，引用具体数据
3. 如果文档中没有相关信息，明确说明
4. 使用中文回答，保持专业术语的准确性

重要原则：
- 只基于提供的文档内容回答，不要编造信息
- 如果信息不足，明确指出缺失的部分
- 涉及财务数据时，要准确引用数字和单位"""
        
        user_prompt = f"""基于以下文档内容，回答用户问题。

文档内容：
{context_texts if context_texts else "未检索到相关文档"}

用户问题: {state['query']}

请提供专业、准确的回答："""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        final_answer = llm.invoke(messages).content
        
        logger.info(f"生成答案完成，长度: {len(final_answer)}")
        
        return {
            "final_answer": final_answer,
        }
        
    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        return {
            "final_answer": f"生成答案时出错: {e}",
        }

