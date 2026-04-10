from src.paper_parser import parse_paper
from src.chunker import build_hierarchical_chunks
from collections import Counter
from src.retriever import build_index, search_with_rerank
def test_chunker(chunks):
    print(f"总 chunk 数: {len(chunks)}")
    print(f"各层分布: {dict(Counter(c['level'] for c in chunks))}")

    # 检查 chunk_id 是否唯一递增（关键验证）
    ids = [c["chunk_id"] for c in chunks]
    print(f"chunk_id 范围: {min(ids)} - {max(ids)}")
    print(f"chunk_id 是否唯一: {len(set(ids)) == len(ids)}")

    # 各层预览
    for level in ["document", "section", "paragraph"]:
        level_chunks = [c for c in chunks if c["level"] == level]
        print(f"\n=== {level} ({len(level_chunks)} 个) ===")
        for c in level_chunks[:3]:
            print(f"  [{c['chunk_id']}] [{c.get('section_title')}] len={len(c['text'])}")
            print(f"    {c['text'][:100]}...")

    # 长度分布
    para_lens = [len(c["text"]) for c in chunks if c["level"] == "paragraph"]
    if para_lens:
        print(f"\n段落级长度: min={min(para_lens)}, max={max(para_lens)}, "
            f"mean={sum(para_lens)//len(para_lens)}")
        
    # 在 test_tool.py 最后加
    short_chunks = [c for c in chunks if len(c["text"]) < 50 and c["level"] == "paragraph"]
    print(f"\n=== 短 chunks ({len(short_chunks)} 个) ===")
    for c in short_chunks[:5]:
        print(f"  [{c['chunk_id']}] len={len(c['text'])}")
        print(f"    repr: {repr(c['text'])}")

def test_retriever(chunks):
    
    print("\n正在 embedding...")
    vectors = build_index(chunks)
    print(f"向量矩阵 shape: {vectors.shape}")
    print(f"应该是: ({len(chunks)}, 384)")   # 384 是 all-MiniLM-L6-v2 的维度

    # 测试一个 P1 类查询（宏观问题）
    print("\n=== 测试 1: P1 类 - 核心贡献 ===")
    results = search_with_rerank(
        "What is the core contribution of this paper?",
        chunks, vectors, recall_k=20, final_k=3
    )
    for i, (c, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f} level={c['level']} section={c['section_title']}")
        print(f"    {c['text'][:120]}...")

    # 测试一个 P2 类查询（细节问题）
    print("\n=== 测试 2: P2 类 - 实验细节 ===")
    results = search_with_rerank(
        "What dataset was used for evaluation?",
        chunks, vectors, recall_k=20, final_k=3
    )
    for i, (c, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f} level={c['level']} section={c['section_title']}")
        print(f"    {c['text'][:120]}...")

if __name__ == "__main__":
    PDF_PATH = "./data/papers/1.2016-Arad_and_Ben_Shahar-Sparse_Recovery_of_Hyperspectral_Signal_from_Natural_RGB_Images.pdf"
    paper = parse_paper(PDF_PATH)
    chunks = build_hierarchical_chunks(paper, paper_id="arad_2016")

    # test_chunker(chunks)
    test_retriever(chunks)