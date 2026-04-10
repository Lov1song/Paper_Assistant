# src/retriever.py
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder

# 全局加载模型（只在 import 时加载一次）
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)

def build_index(chunks: list[dict]) -> np.ndarray:
    """
    把 chunks 的 text 字段编码成向量矩阵。
    返回: shape (N, D) 的 np.ndarray
    """
    # TODO: 你来写
    # 提示：提取所有 text，batch encode，记得 normalize_embeddings=True
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts,batch_size = 32, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)  # shape (N, D)
    return vectors #shape (N, D) 的 np.ndarray

def search(query: str, chunks: list[dict], vectors: np.ndarray, 
           top_k: int = 5) -> list[tuple[dict, float]]:
    """
    向量检索：返回 top_k 个最相关的 chunks 和相似度分数。
    """
    # TODO: 你来写
    # 提示：直接抄你之前的 search 函数，适配 dict 格式
    query_vec = embedder.encode([query],normalize_embeddings=True).astype(np.float32)  # shape (1, D)
    scores = np.dot(vectors,query_vec.T).squeeze()  # shape (N,)
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    results = [(chunks[i],float(scores[i])) for i in top_k_indices]
    return results

def diverse_top_k(candidates, scores, final_k=3):
    """
    从候选池里选 top_k，保证不同 level 的多样性。
    
    策略：第一轮每个出现过的 level 各取最高分的 1 个；
         第二轮按纯分数继续填满 final_k。
    """
    # 把 (chunk, score) 配对并按分数降序排序
    sorted_pairs = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    chosen = []
    chosen_ids = set()   # 用 chunk_id 去重，避免同一个 chunk 被选两次
    levels_seen = set()
    
    # 第一轮：每个 level 挑最高分的一个
    for c, s in sorted_pairs:
        if c["level"] not in levels_seen:
            chosen.append((c, s))
            chosen_ids.add(c["chunk_id"])
            levels_seen.add(c["level"])
            if len(chosen) >= final_k:
                return chosen
    
    # 第二轮：按分数继续填
    for c, s in sorted_pairs:
        if c["chunk_id"] in chosen_ids:
            continue
        chosen.append((c, s))
        chosen_ids.add(c["chunk_id"])
        if len(chosen) >= final_k:
            return chosen
    
    return chosen

def search_with_rerank(query: str, chunks: list[dict], vectors: np.ndarray,
                        recall_k: int = 20, final_k: int = 5) -> list[tuple[dict, float]]:
    """
    两阶段检索 + 分层召回
    确保 document、section、paragraph 三层都有代表进入 rerank 阶段
    """
    query_vector = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores = np.dot(query_vector, vectors.T).squeeze()
    
    # === 新增：分层召回 ===
    # 按 level 分组索引
    doc_indices = [i for i, c in enumerate(chunks) if c["level"] == "document"]
    section_indices = [i for i, c in enumerate(chunks) if c["level"] == "section"]
    para_indices = [i for i, c in enumerate(chunks) if c["level"] == "paragraph"]
    
    # 分别取每层的 top
    def top_from_group(group_indices, k):
        """从一个 level 的索引里取相似度 top-k"""
        if not group_indices:
            return []
        group_scores = [(i, scores[i]) for i in group_indices]
        group_scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in group_scores[:k]]
    
    # 分配配额：document 全取，section 全取，paragraph 取剩余名额
    doc_top = top_from_group(doc_indices, k=len(doc_indices))  # 全部
    section_top = top_from_group(section_indices, k=len(section_indices))  # 全部
    para_quota = recall_k - len(doc_top) - len(section_top)
    para_top = top_from_group(para_indices, k=max(para_quota, 1))
    
    recall_ids = doc_top + section_top + para_top
    candidates = [chunks[i] for i in recall_ids]
    print(f"DEBUG 召回 {len(candidates)} 个候选, level 分布: {dict(Counter(c['level'] for c in candidates))}")
    # === reranker 精排（不变）===
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    return diverse_top_k(candidates, rerank_scores, final_k=final_k)
