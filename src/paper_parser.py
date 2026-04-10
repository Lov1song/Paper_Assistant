# pdf_test.py
import fitz
import re

def light_clean(text: str) -> str:
    """轻清理：统一换行 + 连字符修复 + 行末断词"""
    # 🔑 第一步：统一换行（解决 Windows \r\n 问题）
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 连字符替换
    ligature_map = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff',
        'ﬃ': 'ffi', 'ﬄ': 'ffl',
    }
    for lig, rep in ligature_map.items():
        text = text.replace(lig, rep)
    
    # 行末断词修复
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text

def identify_sections(text: str) -> list[dict]:
    pattern = re.compile(
        r'\n(\d{1,2})\n([A-Z][A-Za-z][A-Za-z\s]{1,48}[A-Za-z])\n',
        re.MULTILINE
    )
    
    raw_matches = list(pattern.finditer(text))
    
    # 后验过滤：章节号必须递增且从 1 开始
    valid_matches = []
    expected_num = 1
    for m in raw_matches:
        num = int(m.group(1))
        if num == expected_num:
            valid_matches.append(m)
            expected_num += 1
    
    sections = []
    for i, match in enumerate(valid_matches):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start = match.end()
        end = valid_matches[i+1].start() if i+1 < len(valid_matches) else len(text)
        content = text[start:end].strip()
        sections.append({
            "number": section_num,
            "title": section_title,
            "content": content
        })
    
    return sections

def remove_references(text: str) -> str:
    """移除参考文献"""
    # 要求 References 前有换行，后面紧跟 "1."（第一条引用）
    pattern = re.compile(r'\nReferences\n+1\.\s', re.DOTALL)
    match = pattern.search(text)
    if match:
        return text[:match.start()]
    return text

def extract_abstract(text: str) -> str:
    """提取摘要：从 'Abstract.' 到 '1\nIntroduction' 之间"""
    pattern = re.compile(
        r'Abstract\.\s*(.*?)\n\d+\n[A-Z]',
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

def parse_paper(pdf_path: str) -> dict:
    """
    解析一篇论文，返回结构化信息
    """
    doc = fitz.open(pdf_path)
    
    # 1. 提取所有文本
    raw_text = ""
    for page in doc:
        raw_text += page.get_text() + "\n\n"
    
    # 2. 提取元数据
    metadata = {
        "filename": pdf_path.split("/")[-1].split("\\")[-1],
        "num_pages": len(doc),
    }
    doc.close()
    
    # 3. 清理
    cleaned = light_clean(raw_text)
    
    # 4. 切掉参考文献
    body = remove_references(cleaned)
    
    # 5. 提取 abstract
    abstract = extract_abstract(body)
    
    # 6. 识别章节
    sections = identify_sections(body)
    
    return {
        "metadata": metadata,
        "abstract": abstract,
        "sections": sections,
        "full_text": body,  # 留一份完整文本，万一章节识别失败有兜底
    }
 
if __name__ == "__main__":
    PDF_PATH = "./data/papers/1.2016-Arad_and_Ben_Shahar-Sparse_Recovery_of_Hyperspectral_Signal_from_Natural_RGB_Images.pdf"
    paper = parse_paper(PDF_PATH)

    print(f"文件: {paper['metadata']['filename']}")
    print(f"页数: {paper['metadata']['num_pages']}")

    print(f"\n--- Abstract ({len(paper['abstract'])} 字符) ---")
    print(paper['abstract'][:500])

    print(f"\n--- 识别到 {len(paper['sections'])} 个章节 ---")
    for s in paper['sections']:
        print(f"  [{s['number']}] {s['title']}: {len(s['content'])} 字符")

    print(f"\n--- 正文总长度: {len(paper['full_text'])} 字符 ---")
    print(f"切掉了: {42602 - len(paper['full_text'])} 字符")