# src/tools.py
from src.retriever import search_with_rerank
def search_paper(query: str,CHUNKS,VECTORS) -> str:
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

def list_papers(CHUNKS) -> str:
    """列出所有已加载的论文"""
    paper_info = {}
    for c in CHUNKS:
        pid = c["paper_id"]
        if pid not in paper_info:
            paper_info[pid] = {
                "title": c["paper_title"],
                "chunk_count": 0,
                "has_sections": False,
            }
        paper_info[pid]["chunk_count"] += 1
        if c["level"] == "section":
            paper_info[pid]["has_sections"] = True

    lines = [f"已加载 {len(paper_info)} 篇论文：\n"]
    for i, (pid, info) in enumerate(sorted(paper_info.items()), 1):
        structure = "有章节结构" if info["has_sections"] else "仅段落级"
        lines.append(f"{i}. {info['title']} ({info['chunk_count']} chunks, {structure})")

    return "\n".join(lines)

def get_paper_abstract(paper_name: str,CHUNKS) -> str:
    """获取指定论文的摘要"""
    # 在 chunks 里找 level=document 的 chunk
    for c in CHUNKS:
        if c["level"] == "document" and paper_name.lower() in c["paper_id"].lower():
            return f"论文: {c['paper_title']}\n\n摘要:\n{c['text']}"

    # 模糊匹配：用户可能只输入了部分名字
    candidates = []
    for c in CHUNKS:
        if c["level"] == "document":
            if any(word.lower() in c["paper_id"].lower() for word in paper_name.split()):
                candidates.append(c)

    if candidates:
        results = []
        for c in candidates:
            results.append(f"论文: {c['paper_title']}\n摘要:\n{c['text']}")
        return "\n\n---\n\n".join(results)

    return f"未找到包含 '{paper_name}' 的论文。请使用 list_papers 查看所有论文。"

def compare_papers(paper_a: str, paper_b: str,aspect,CHUNKS) -> str:
    """检索两篇论文的相关内容，便于对比分析"""
    results_a = []
    results_b = []

    for c in CHUNKS:
        pid = c["paper_id"].lower()
        if paper_a.lower() in pid:
            results_a.append(c)
        if paper_b.lower() in pid:
            results_b.append(c)

    if not results_a:
        return f"未找到包含 '{paper_a}' 的论文。"
    if not results_b:
        return f"未找到包含 '{paper_b}' 的论文。"

    # 如果指定了对比维度，用 search 检索相关内容
    if aspect:
        # 在论文 A 的 chunks 里检索
        a_chunks = [c for c in CHUNKS if paper_a.lower() in c["paper_id"].lower()]
        b_chunks = [c for c in CHUNKS if paper_b.lower() in c["paper_id"].lower()]

        output = f"=== 论文 A: {results_a[0]['paper_title']} ===\n"
        # 找 abstract
        a_abstract = [c for c in a_chunks if c["level"] == "document"]
        if a_abstract:
            output += f"摘要: {a_abstract[0]['text'][:500]}\n"

        output += f"\n=== 论文 B: {results_b[0]['paper_title']} ===\n"
        b_abstract = [c for c in b_chunks if c["level"] == "document"]
        if b_abstract:
            output += f"摘要: {b_abstract[0]['text'][:500]}\n"

        return output

    # 没指定维度，返回两篇的 abstract
    output = f"=== 论文 A: {results_a[0]['paper_title']} ===\n"
    a_abs = [c for c in results_a if c["level"] == "document"]
    output += f"摘要: {a_abs[0]['text'][:500]}\n" if a_abs else "（未找到摘要）\n"

    output += f"\n=== 论文 B: {results_b[0]['paper_title']} ===\n"
    b_abs = [c for c in results_b if c["level"] == "document"]
    output += f"摘要: {b_abs[0]['text'][:500]}\n" if b_abs else "（未找到摘要）\n"

    return output