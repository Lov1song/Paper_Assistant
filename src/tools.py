# src/tools.py
from retriever import search_with_rerank

def search_paper(query: str) -> str:
    results = search_with_rerank(query, CHUNKS, VECTORS, recall_k=20, final_k=5)
    
    if not results:
        return "未找到相关内容。"
    
    formatted_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        paper = chunk.get("paper_title", "未知论文")
        section = chunk.get("section_title", "未知章节")
        level = chunk["level"]
        text = chunk["text"]
        formatted_parts.append(
            f"[结果 {i}] (论文: {paper} | 来源: {section} | 类型: {level} | 相关度: {score:.3f})\n{text}"
        )
    
    return "\n\n---\n\n".join(formatted_parts)