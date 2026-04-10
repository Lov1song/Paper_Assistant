# src/tools.py
from retriever import search_with_rerank

def search_paper(query: str, chunks, vectors) -> str:
    """
    在论文中检索与 query 相关的内容。
    返回格式化的检索结果（字符串），供 agent 使用。
    """
    
    results = search_with_rerank(query, chunks, vectors, recall_k=20, final_k=5)
    
    if not results:
        return "未找到相关内容。"
    
    formatted_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        section = chunk.get("section_title", "未知章节")
        level = chunk["level"]
        text = chunk["text"]
        
        formatted_parts.append(
            f"[结果 {i}] (来源: {section} | 类型: {level} | 相关度: {score:.3f})\n{text}"
        )
    
    return "\n\n---\n\n".join(formatted_parts)