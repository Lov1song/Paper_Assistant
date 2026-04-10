import os
import glob
import pickle
from src.paper_parser import parse_paper
from src.chunker import build_hierarchical_chunks
from src.retriever import build_index

INDEX_DIR = "data/index"
CACHE_FILE = os.path.join(INDEX_DIR, "all_papers.pkl")

def load_all_papers(papers_dir: str = "data/papers") -> tuple[list[dict], any]:
    """
    加载所有论文。优先读缓存，缓存过期则重新构建。
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    pdf_files = sorted(glob.glob(os.path.join(papers_dir, "*.pdf")))
    
    if not pdf_files:
        print(f"⚠️ {papers_dir} 目录下没有找到 PDF 文件")
        return [], None
    
    # 检查缓存是否有效
    if _is_cache_valid(pdf_files):
        print("发现有效缓存，直接加载...")
        return _load_cache()
    
    # 缓存无效，重新构建
    print("未发现缓存或论文有更新，重新构建索引...")
    all_chunks = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        paper_id = filename.replace(".pdf", "")
        
        print(f"\n  正在解析: {filename}")
        paper = parse_paper(pdf_path)
        print(f"    页数: {paper['metadata']['num_pages']}, 章节: {len(paper['sections'])} 个")
        
        chunks = build_hierarchical_chunks(paper, paper_id=paper_id)
        print(f"    生成 {len(chunks)} 个 chunks")
        
        all_chunks.extend(chunks)
    
    print(f"\n正在 embedding（共 {len(all_chunks)} 个 chunks）...")
    vectors = build_index(all_chunks)
    
    # 保存缓存
    _save_cache(all_chunks, vectors, pdf_files)
    print("索引已缓存到磁盘")
    
    return all_chunks, vectors


def _is_cache_valid(pdf_files: list[str]) -> bool:
    """检查缓存是否存在且没有过期"""
    if not os.path.exists(CACHE_FILE):
        return False
    
    # 如果任何 PDF 比缓存文件更新，缓存就过期了
    cache_time = os.path.getmtime(CACHE_FILE)
    for pdf in pdf_files:
        if os.path.getmtime(pdf) > cache_time:
            return False
    
    # 检查 PDF 数量是否一致（新增或删除了论文）
    try:
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if set(cached["pdf_files"]) != set(os.path.basename(p) for p in pdf_files):
            return False
    except Exception:
        return False
    
    return True


def _save_cache(chunks, vectors, pdf_files):
    """保存缓存"""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "vectors": vectors,
            "pdf_files": [os.path.basename(p) for p in pdf_files],
        }, f)


def _load_cache() -> tuple[list[dict], any]:
    """读取缓存"""
    with open(CACHE_FILE, "rb") as f:
        cached = pickle.load(f)
    chunks = cached["chunks"]
    vectors = cached["vectors"]
    print(f"  已加载 {len(chunks)} 个 chunks")
    return chunks, vectors