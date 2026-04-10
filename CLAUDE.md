# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

基于 RAG 的学术论文助手，读取 PDF 研究论文、提取并清洗文本、建立向量索引，通过 Agent 回答问题。

## 运行方式

```bash
pip install -r requirements.txt

# CLI 交互式问答（主要入口）
python cli.py

# 集成测试（单篇论文的 parse → chunk → embed → rerank）
python test_tool.py

# 以下入口尚未实现
python app.py
```

## 架构与实现状态

流水线：PDF → `parse_paper` → `build_hierarchical_chunks` → `build_index` → `search_with_rerank` → Agent

```
src/
  paper_parser.py    — parse_paper()，PDF 解析与文本清洗（已完成）
  chunker.py         — build_hierarchical_chunks()，三层 chunk 构建（已完成）
  retriever.py       — build_index() / search_with_rerank()（已完成）
  load_all_paper.py  — 批量加载多篇 PDF，带磁盘缓存（已完成）
  tools.py           — search_paper / list_papers / get_paper_abstract / compare_papers（已完成）
  agent.py           — LLM Agent 编排（未实现）
cli.py               — CLI 交互式问答主入口（已完成）
app.py               — Web/GUI 主入口（未实现）
test_tool.py         — 集成测试：parse → chunk → embed → rerank（测试单篇论文）
data/
  papers/            — 放 PDF 的目录
  index/all_papers.pkl — 向量索引缓存（自动生成）
```

## Chunk 结构

```python
{
    "text": str,
    "level": "document" | "section" | "paragraph",
    "paper_id": str,          # 论文唯一标识（文件名去掉 .pdf）
    "paper_title": str,
    "section_number": str,    # document 级为 None
    "section_title": str,
    "chunk_id": int,          # 全局递增
}
```

三层语义：
- `document`：Abstract，服务宏观问题（"这篇论文讲什么"）
- `section`：整章内容，服务章节级问题
- `paragraph`：按 400 字符切分的段落，服务细节查询

无章节结构时（正则匹配失败），退化为两层：document + paragraph from full_text。

## 检索流程（retriever.py）

- 向量模型：`all-MiniLM-L6-v2`（384 维，模块 import 时全局加载）
- 重排模型：`BAAI/bge-reranker-base`（同上）
- `search_with_rerank()`：分层配额召回（20% document / 30% section / 50% paragraph）→ reranker 精排 → `diverse_top_k()` 保证三层各有代表

## PDF 解析（paper_parser.py）

`parse_paper(pdf_path)` 返回：
```python
{
    "metadata": {"filename": str, "num_pages": int},
    "abstract": str,
    "sections": [{"number": str, "title": str, "content": str}],
    "full_text": str,   # 章节识别失败时的兜底
}
```

关键实现细节：
- `light_clean()`：LaTeX 连字符替换（ﬁ→fi 等）、行末断词还原、换行统一
- `identify_sections()`：正则匹配 `\n数字\n标题\n` 模式，要求章节号从 1 递增——对非标准格式论文可能失效
- `remove_references()`：截断到 `References\n1.` 之前

## 批量索引缓存（load_all_paper.py）

`load_all_papers(papers_dir)` 自动检查 `data/index/all_papers.pkl` 缓存有效性（比对 PDF 修改时间和文件列表），有效则直接加载，否则重新解析所有 PDF 并构建向量索引后保存。

## 已知问题 / 待解决

- 超长 section chunk（>13000 字符）超过 embedding 模型 max_length，暂未截断
- `is_junk_paragraph()` 中的垃圾过滤规则（LNCS/Springer 等）为硬编码，多来源论文时可能需要扩展
- `app.py` 尚未实现（预期为 Web/GUI 界面）
- `agent.py` 为空（CLI 中 Agent 循环直接写在 `cli.py` 的 `run_agent()` 里，未抽离）
- API Key 通过 `.env` 文件配置（参考 `.env.example`），使用 DeepSeek API（`DEEPSEEK_API_KEY`）
