# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

基于 RAG 的学术论文助手，读取 PDF 研究论文、提取并清洗文本、建立向量索引，通过 Agent 回答问题。

## 运行方式

```bash
pip install -r requirements.txt

# 当前主要测试入口（chunker + retriever 的集成测试）
python test_tool.py

# PDF 解析参考实现（独立可运行）
python pdf_test.py

# 以下入口尚未实现
python app.py
python cli.py
```

## 架构与实现状态

流水线：PDF → parse_paper → build_hierarchical_chunks → build_index → search_with_rerank → Agent

```
pdf_test.py          — parse_paper() 参考实现（已完成，待迁移）
src/
  paper_parser.py    — 已经构建完成
  chunker.py         — build_hierarchical_chunks()，三层 chunk 构建（已完成）
  retriever.py       — build_index() / search_with_rerank()（已完成）
  tools.py           — LangChain 工具定义（未实现）
  agent.py           — LLM Agent 编排（未实现）
  pdf_loader.py      — 批量加载多篇 PDF（未实现）
test_tool.py         — 当前集成测试：parse → chunk → embed → rerank
```

## Chunk 结构

每个 chunk 是一个 dict：

```python
{
    "text": str,
    "level": "document" | "section" | "paragraph",
    "paper_id": str,          # 论文唯一标识
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

## 检索流程（retriever.py）

- 向量模型：`all-MiniLM-L6-v2`（384 维，模块 import 时全局加载）
- 重排模型：`BAAI/bge-reranker-base`（同上）
- `search_with_rerank()`：分层召回（document 全取、section 全取、paragraph 按配额）→ reranker 精排 → `diverse_top_k()` 保证三层都有代表

## PDF 解析（pdf_test.py → 待迁移至 paper_parser.py）

`parse_paper()` 返回：
```python
{
    "metadata": {"filename": str, "num_pages": int},
    "abstract": str,
    "sections": [{"number": str, "title": str, "content": str}],
    "full_text": str,   # 章节识别失败时的兜底
}
```

文本清洗（`light_clean()`）：LaTeX 连字符替换（ﬁ→fi 等）、行末断词还原、换行统一。章节识别依赖正则匹配"递增数字 + 标题"模式，对非标准格式论文可能失效。

## 已知问题 / 待解决

- `src/paper_parser.py` 为空，`test_tool.py` 中的 `from src.paper_parser import parse_paper` 实际上无法运行（需将 pdf_test.py 的代码迁移过去）
- 超长 section chunk（>13000 字符）超过 embedding 模型 max_length，暂未处理
- `chunker.py` 中 `is_junk_paragraph()` 硬编码了论文 1 的页眉特征，多论文时需泛化
