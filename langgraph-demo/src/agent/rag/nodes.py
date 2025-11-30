"""图RAG节点实现"""

import logging
import os
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


def decompose_query(state: GraphRAGState) -> Dict[str, Any]:
    """查询分解节点：将复杂查询分解成多个子查询
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    query = state['query']
    logger.info(f"开始分解查询: {query}")
    
    try:
        llm = get_llm()
        
        decompose_prompt = f"""你是一个专业的查询分解专家。请将用户的复杂查询分解成多个独立的子查询。

用户查询: {query}

分解规则：
1. 识别查询中的多个实体（公司、产品、概念等）
2. 识别查询中的多个维度（背景、财报、业务、支出、收入、战略、竞争等）
3. 为每个实体和每个维度的组合生成一个独立的子查询
4. 子查询应该是完整的、可以独立检索的问题
5. 不要包含原查询的完整内容，只生成分解后的子查询
6. 子查询应该具体、明确，便于检索

示例：
- 原查询："京东和美团和阿里外卖大战"
- 子查询：
京东的背景
京东的财报
京东的外卖支出
美团的背景
美团的财报
美团的外卖支出
阿里的背景
阿里的财报
阿里的外卖支出

重要：只返回子查询列表，每行一个子查询，不要编号，不要其他说明，不要添加任何前缀或后缀。

子查询列表：
"""
        
        response = llm.invoke([HumanMessage(content=decompose_prompt)]).content.strip()
        
        # 解析子查询
        sub_queries = []
        for line in response.split('\n'):
            line = line.strip()
            # 移除编号（如 "1. ", "1) " 等）
            if line:
                # 移除开头的数字和标点
                import re
                line = re.sub(r'^[\d\.\)\s]+', '', line)
                if line:
                    sub_queries.append(line)
        
        # 如果没有分解出子查询，使用原查询
        if not sub_queries:
            logger.warning("未能分解查询，使用原查询")
            sub_queries = [query]
        
        logger.info(f"查询分解完成，生成 {len(sub_queries)} 个子查询:")
        for i, sq in enumerate(sub_queries, 1):
            logger.info(f"  子查询 {i}: {sq}")
        
        return {
            "sub_queries": sub_queries,
            "sub_query_results": {},
        }
        
    except Exception as e:
        logger.error(f"查询分解失败: {e}")
        # 分解失败时使用原查询
        return {
            "sub_queries": [query],
            "sub_query_results": {},
        }


def retrieve(state: GraphRAGState) -> Dict[str, Any]:
    """检索节点：从向量数据库检索相关文档（支持子查询）
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    # 检查是否有子查询需要处理
    sub_queries = state.get('sub_queries', [])
    sub_query_results = state.get('sub_query_results', {})
    
    # 如果有子查询且还未全部检索完，处理子查询
    if sub_queries and len(sub_query_results) < len(sub_queries):
        # 找到还未检索的子查询
        remaining_queries = [sq for sq in sub_queries if sq not in sub_query_results]
        if remaining_queries:
            current_query = remaining_queries[0]
            logger.info(f"开始检索子查询 [{len(sub_query_results)+1}/{len(sub_queries)}]: {current_query}")
        else:
            # 所有子查询都已检索，进入汇总阶段
            return aggregate_results(state)
    else:
        # 没有子查询，使用原查询
        current_query = state['query']
        logger.info(f"开始检索，查询: {current_query}")
    
    try:
        # 生成查询向量
        embedding_model = get_embedding()
        query_embedding = embedding_model.embed_query(current_query)
        
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
        
        logger.info(f"向量检索返回 {len(search_results)} 个结果")
        for i, result in enumerate(search_results[:5]):  # 只记录前5个
            logger.info(f"结果 {i+1}: 相似度={result.score:.4f}, 阈值={config.rag.similarity_threshold}")
        
        for result in search_results:
            if result.score >= config.rag.similarity_threshold:
                retrieved_chunks.append({
                    "content": result.chunk.content,
                    "metadata": result.chunk.metadata,
                    "score": result.score,
                    "sub_query": current_query,  # 标记来源子查询
                })
                retrieval_scores.append(result.score)
        
        logger.info(f"经过阈值过滤后，检索到 {len(retrieved_chunks)} 个相关文档块")
        
        # 如果没有结果且阈值过滤太严格，降低阈值重试
        if len(retrieved_chunks) == 0 and len(search_results) > 0:
            logger.warning(f"阈值 {config.rag.similarity_threshold} 过滤掉了所有结果，使用动态阈值")
            # 使用动态阈值：取最高分的75%或0.3，取较大值（更宽松）
            if search_results:
                max_score = max(r.score for r in search_results)
                # 动态阈值：最高分的75%，但至少0.3（更宽松，提高召回率）
                dynamic_threshold = max(max_score * 0.75, 0.3)
                logger.info(f"最高分: {max_score:.4f}, 使用动态阈值: {dynamic_threshold:.4f}")
                
                for result in search_results:
                    if result.score >= dynamic_threshold:
                        retrieved_chunks.append({
                            "content": result.chunk.content,
                            "metadata": result.chunk.metadata,
                            "score": result.score,
                            "sub_query": current_query,  # 标记来源子查询
                        })
                        retrieval_scores.append(result.score)
                
                logger.info(f"使用动态阈值后，检索到 {len(retrieved_chunks)} 个相关文档块")
        
        # 如果有子查询，保存当前子查询的结果
        if sub_queries:
            sub_query_results[current_query] = retrieved_chunks
            # 检查是否所有子查询都已检索完
            # 保存当前子查询的检索结果
            sub_query_results[current_query] = retrieved_chunks
            
            if len(sub_query_results) >= len(sub_queries):
                # 所有子查询检索完成，进入汇总
                logger.info(f"所有子查询检索完成，准备汇总结果")
                return {
                    "sub_query_results": sub_query_results,
                }
            else:
                # 还有子查询未检索，继续检索下一个
                return {
                    "sub_query_results": sub_query_results,
                }
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "retrieval_scores": retrieval_scores,
            "use_web_search": len(retrieved_chunks) == 0,  # 如果没有检索到结果，标记需要网络搜索
        }
        
    except Exception as e:
        logger.error(f"检索失败: {e}")
        # 如果有子查询，标记当前子查询检索失败
        if sub_queries:
            sub_query_results[current_query] = []
            if len(sub_query_results) >= len(sub_queries):
                return aggregate_results(state)
            return {
                "sub_query_results": sub_query_results,
            }
        return {
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "use_web_search": True,  # 检索失败时也需要网络搜索
        }


def aggregate_results(state: GraphRAGState) -> Dict[str, Any]:
    """汇总节点：汇总所有子查询的检索结果
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    sub_queries = state.get('sub_queries', [])
    sub_query_results = state.get('sub_query_results', {})
    
    if not sub_queries or not sub_query_results:
        # 没有子查询，直接返回
        return {}
    
    logger.info(f"开始汇总 {len(sub_queries)} 个子查询的检索结果")
    
    # 汇总所有子查询的结果
    all_chunks = []
    all_scores = []
    
    for sub_query, chunks in sub_query_results.items():
        logger.info(f"子查询 '{sub_query}' 检索到 {len(chunks)} 个文档块")
        all_chunks.extend(chunks)
        all_scores.extend([chunk.get('score', 0.0) for chunk in chunks])
    
    # 去重：基于内容相似度去重（简单版本：基于内容hash）
    seen_content = set()
    unique_chunks = []
    unique_scores = []
    
    for chunk in all_chunks:
        content_hash = hash(chunk['content'][:100])  # 使用前100字符作为hash
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_chunks.append(chunk)
            unique_scores.append(chunk.get('score', 0.0))
    
    logger.info(f"汇总完成：原始 {len(all_chunks)} 个文档块，去重后 {len(unique_chunks)} 个文档块")
    
    return {
        "retrieved_chunks": unique_chunks,
        "retrieval_scores": unique_scores,
        "use_web_search": len(unique_chunks) == 0,  # 如果汇总后没有结果，标记需要网络搜索
    }


def rerank(state: GraphRAGState) -> Dict[str, Any]:
    """重排序节点：使用语义模型对检索结果进行重排序，提升相关性
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    retrieved_chunks = state.get('retrieved_chunks', [])
    
    if not retrieved_chunks or not config.rag.enable_reflection:
        # 如果没有检索结果或禁用重排序，直接返回
        return {
            "reranked_chunks": retrieved_chunks,
        }
    
    logger.info(f"开始重排序，输入 {len(retrieved_chunks)} 个文档块")
    
    try:
        llm = get_llm()
        query = state['query']
        
        # 构建重排序提示词
        chunks_text = "\n\n".join([
            f"[文档 {i+1}] (相似度: {chunk.get('score', 0):.4f})\n{chunk['content'][:500]}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        rerank_prompt = f"""你是一个专业的文档相关性评估专家。请根据用户问题，对以下检索到的文档块进行重排序，选出最相关的文档。

用户问题: {query}

检索到的文档块:
{chunks_text}

请评估每个文档与问题的相关性，并返回最相关的文档索引（从1开始），按相关性从高到低排列。
只返回数字，用逗号分隔，例如：3,1,5,2,4

最相关的文档索引（按相关性排序）："""
        
        response = llm.invoke([HumanMessage(content=rerank_prompt)]).content.strip()
        
        # 解析返回的索引
        try:
            indices = [int(x.strip()) - 1 for x in response.split(',') if x.strip().isdigit()]
            # 过滤有效索引
            valid_indices = [i for i in indices if 0 <= i < len(retrieved_chunks)]
            
            if valid_indices:
                # 按重排序后的顺序排列
                reranked = [retrieved_chunks[i] for i in valid_indices]
                # 添加剩余未排序的文档
                remaining_indices = set(range(len(retrieved_chunks))) - set(valid_indices)
                reranked.extend([retrieved_chunks[i] for i in remaining_indices])
            else:
                # 如果解析失败，保持原顺序
                reranked = retrieved_chunks
        except Exception as e:
            logger.warning(f"解析重排序结果失败: {e}，保持原顺序")
            reranked = retrieved_chunks
        
        # 只保留前 rerank_top_k 个
        reranked = reranked[:config.rag.rerank_top_k]
        
        logger.info(f"重排序完成，保留 {len(reranked)} 个最相关的文档块")
        
        return {
            "reranked_chunks": reranked,
        }
        
    except Exception as e:
        logger.error(f"重排序失败: {e}")
        return {
            "reranked_chunks": retrieved_chunks[:config.rag.rerank_top_k],
        }


def web_search(state: GraphRAGState) -> Dict[str, Any]:
    """网络搜索节点：使用Tavily进行联网搜索
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    query = state['query']
    use_web_search = state.get('use_web_search', False)
    
    if not use_web_search:
        logger.info("跳过网络搜索（知识库已有足够信息）")
        return {
            "web_search_results": [],
        }
    
    logger.info(f"开始网络搜索: {query}")
    
    try:
        # 尝试导入Tavily
        try:
            from tavily import TavilyClient
        except ImportError as e:
            logger.error(f"【web_search】Tavily未安装: {e}，请运行：pip install tavily-python")
            logger.error(f"【web_search】如果已安装，请运行：pip install -e . 重新安装项目依赖")
            return {
                "web_search_results": [],
            }
        
        # 获取API密钥
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.warning("TAVILY_API_KEY未设置，跳过网络搜索")
            return {
                "web_search_results": [],
            }
        
        # 初始化Tavily客户端
        client = TavilyClient(api_key=tavily_api_key)
        
        # 执行搜索
        response = client.search(
            query=query,
            search_depth="advanced",  # 使用高级搜索
            max_results=5,  # 最多返回5个结果
        )
        
        # 解析搜索结果
        web_results = []
        if response and 'results' in response:
            for result in response['results']:
                web_results.append({
                    "title": result.get('title', ''),
                    "url": result.get('url', ''),
                    "content": result.get('content', ''),
                    "score": result.get('score', 0.0),
                })
        
        logger.info(f"网络搜索完成，找到 {len(web_results)} 个结果")
        
        return {
            "web_search_results": web_results,
        }
        
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return {
            "web_search_results": [],
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
        
        # 优先使用重排序后的结果，如果没有则使用原始检索结果
        chunks_to_use = state.get('reranked_chunks', []) or state.get('retrieved_chunks', [])
        
        # 构建上下文（知识库内容）
        context_texts = "\n\n".join([
            f"[知识库文档 {i+1}] (相似度: {chunk.get('score', 0):.4f})\n{chunk['content']}\n"
            for i, chunk in enumerate(chunks_to_use)
        ])
        
        # 构建网络搜索结果上下文
        web_results = state.get('web_search_results', [])
        web_context = ""
        if web_results:
            web_context = "\n\n=== 网络搜索结果 ===\n\n"
            web_context += "\n\n".join([
                f"[网络来源 {i+1}] {result.get('title', '')}\nURL: {result.get('url', '')}\n内容: {result.get('content', '')[:500]}\n"
                for i, result in enumerate(web_results)
            ])
        
        # 获取子查询信息（如果有）
        sub_queries = state.get('sub_queries', [])
        sub_query_info = ""
        if sub_queries:
            sub_query_info = f"\n\n=== 查询分解信息 ===\n原查询已分解为 {len(sub_queries)} 个子查询：\n"
            for i, sq in enumerate(sub_queries, 1):
                sub_query_info += f"{i}. {sq}\n"
        
        # 构建系统提示词（金融领域特化，要求生成投资意见）
        system_prompt = """你是一个专业的金融投资顾问AI助手，擅长分析企业财报、投融资报告等金融文档。

你的职责：
1. 基于提供的文档内容和网络搜索结果，准确回答用户问题
2. 回答要专业、准确，引用具体数据
3. 综合分析所有信息（包括从多个子查询检索到的信息），给出专业的投资建议
4. 使用中文回答，保持专业术语的准确性

重要原则：
- 优先使用知识库文档内容，网络搜索结果作为补充
- 如果信息不足，明确指出缺失的部分
- 涉及财务数据时，要准确引用数字和单位
- 综合分析所有子查询的结果，形成完整的答案
- 最后必须提供一条明确的投资意见或建议"""
        
        user_prompt = f"""基于以下信息，回答用户问题并提供投资建议。
{sub_query_info}
=== 知识库文档内容 ===
{context_texts if context_texts else "未检索到相关文档"}
{web_context}

用户问题: {state['query']}

请按以下格式回答：
1. 首先综合分析所有检索到的信息，回答用户的具体问题
2. 引用相关的数据和分析（如果涉及多个实体，分别说明）
3. 最后提供一条明确的投资意见或建议

回答："""
        
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

