"""
文档chunks的构建逻辑：
{
    "text": "实际 chunk 内容...",
    "level": "document" | "section" | "paragraph",
    "paper_id": "arad_2016",          # 论文唯一标识（为 Day 4 多论文做准备）
    "paper_title": "Sparse Recovery of Hyperspectral Signal...",
    "section_number": "3",            # 所属章节号（document 级为 None）
    "section_title": "Hyperspectral Prior for Natural Images",
    "chunk_id": 12,                    # 全局递增 id
}
"""
import fitz
import re

def is_junk_paragraph(para: str) -> bool:
    """判断段落是否为垃圾内容（页眉、版权、元信息等）"""
    para_lower = para.lower()
    
    # 出版元信息
    if 'doi:' in para_lower or 'doi.org' in para_lower:
        return True
    if 'lncs' in para_lower and 'pp.' in para_lower:
        return True
    if 'springer' in para_lower and len(para) < 200:
        return True
    
    # 硬编码的页眉（这篇论文专用）
    if 'hyperspectral signal' in para_lower and len(para) < 150:
        return True
    if 'arad' in para_lower and 'ben-shahar' in para_lower and len(para) < 200:
        return True
    
    return False

def make_chunk(text, level, chunk_id, paper_id, paper_title, 
               section_number=None, section_title=None):
    return {
        "text": text,
        "level": level,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "section_number": section_number,
        "section_title": section_title,
        "chunk_id": chunk_id,
    }

def build_hierarchical_chunks(paper: dict, paper_id: str) -> list[dict]:
    """
    把 parse_paper 的输出转换成三层 chunks。
    
    输入：parse_paper 返回的 dict
    输出：list[dict]，每个元素是一个 chunk
    """
    chunks = []
    chunk_id = 0
    paper_title = paper["metadata"]["filename"].replace(".pdf", "")
    
    # ===== Level 1: 文档级（Abstract）=====
    # TODO: 你来写
    # 一个 chunk，level="document"，text=paper["abstract"]
    # section_number=None, section_title="Abstract"
    chunks.append(
        make_chunk(
            text=paper["abstract"],
            level="document",
            chunk_id=chunk_id,
            paper_id=paper_id,
            paper_title=paper_title,
            section_number=None,
            section_title="Abstract",
        )
    )
    chunk_id += 1
    
    # ===== Level 2: 章节级 =====
    # TODO: 你来写
    # 对 paper["sections"] 里的每个 section：
    # 创建一个 chunk，level="section"，text=section["content"]
    # 注意：有些章节可能很长（13000 字符），超过 embedding 模型的 max_length
    #       暂时不管，先让它跑通。超长的问题我们后面解决。
    sections = paper["sections"]
    for section in sections:
        chunks.append(
            make_chunk(
                text=section["content"],
                level="section",
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                section_number=section["number"],
                section_title=section["title"],
            )
        )
        chunk_id += 1
    


    # ===== Level 3: 段落级 =====
    # TODO: 你来写
    # 对每个 section，把 content 按段落切分
    # 复用你之前写的 smart_split 的逻辑，但简化：
    #   - 按 \n\n 切段落
    #   - 小段落合并到 ~400 字符
    #   - 超过 600 字符的按句子切
    # 每个小 chunk level="paragraph"
    for section in paper["sections"]:
        paragraphs = section["content"].split("\n\n")
        for para in paragraphs:
            current = ""  # ← 每轮开始时重置，防止污染
            para = para.strip()
            if len(para) < 100:   # 只打印短的，避免输出太多
                print(f"DEBUG len={len(para)}, repr={repr(para[:80])}")
            if len(para) < 50:    # ← 跳过过短段落
                continue
            if is_junk_paragraph(para):
                continue
            elif len(para) < 400:
                chunks.append(
                    make_chunk(
                        text=para,
                        level="paragraph",
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        section_number=section["number"],
                        section_title=section["title"],
                    )
                )
                chunk_id += 1
            
            elif len(para) > 600:
                sentences = re.split(r'(?<=[.!?]) +', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) <= 400:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current and len(current) >= 50:  # 跳过过短的 chunk
                            chunks.append(
                                make_chunk(
                                    text=current,
                                    level="paragraph",
                                    chunk_id=chunk_id,
                                    paper_id=paper_id,
                                    paper_title=paper_title,
                                    section_number=section["number"],
                                    section_title=section["title"],
                                ))
                            chunk_id += 1
                        current = sent
                # 收尾：循环结束后还有未保存的内容
                if current and len(current) >= 50:  # 跳过过短的 chunk
                    chunks.append(
                        make_chunk(
                            text=current,
                            level="paragraph",
                            chunk_id=chunk_id,
                            paper_id=paper_id,
                            paper_title=paper_title,
                            section_number=section["number"],
                            section_title=section["title"],
                        ))
                    chunk_id += 1
            else:
                chunks.append(
                    make_chunk( 
                    text=para,
                    level="paragraph",
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    section_number=section["number"],
                    section_title=section["title"],
                ))
                chunk_id += 1
    return chunks
