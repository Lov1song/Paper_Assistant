# 📄 Paper Assistant - 学术论文智能问答助手

一个基于 RAG（Retrieval-Augmented Generation）的学术论文阅读助手。上传 PDF 论文，用自然语言提问，获得基于论文内容的准确回答。

## 效果演示

```
论文助手已就绪，你可以开始提问了

你: 这篇论文是什么方向的？

助手: 根据论文的摘要和引言部分，这篇论文是计算机视觉和计算成像方向的研究，
     具体属于高光谱成像（Hyperspectral Imaging）领域。论文提出了一种低成本、
     快速的方法，直接从RGB图像恢复高质量的高光谱图像...

你: 告诉我如何复现

  [检索中] query: reproduce implementation code
  [检索中] query: algorithm steps procedure
  [检索中] query: sparse dictionary method hyperspectral reconstruction

助手: 根据论文的 Implementation and Results 部分，复现该方法需要以下步骤：
     1. 数据准备：获取高光谱数据库，将光谱范围处理为31个波段...
     2. 字典构建：使用K-SVD算法构建过完备的高光谱字典...
     ...
```

## 核心特性

- **PDF 自动解析**：从 PDF 中提取文本，识别论文结构（Abstract、章节、参考文献）
- **层次化 RAG**：三层索引（文档级 / 章节级 / 段落级），宏观问题和细节问题都能精准回答
- **两阶段检索**：Embedding 召回 + Cross-Encoder 重排，兼顾速度与精度
- **分层召回 + 多样性重排**：解决不同粒度 chunk 数量不平衡导致的检索偏差
- **Function Calling Agent**：基于 DeepSeek API，自动调用检索工具，支持多轮检索
- **中文交互**：用中文提问英文论文，Agent 自动翻译 query 进行检索

## 技术架构

```
PDF论文
  │
  ▼
[paper_parser.py] PDF解析 + 结构识别（PyMuPDF + 正则）
  │  · LaTeX 连字符修复（ﬁ → fi）
  │  · 行末断词修复（hin-\ndered → hindered）
  │  · 章节标题识别 + Abstract 提取 + 参考文献切除
  │
  ▼
[chunker.py] 层次化切分 + 脏数据过滤
  │  · Level 1: 文档级（Abstract）
  │  · Level 2: 章节级（每章完整内容）
  │  · Level 3: 段落级（~300-400 字符）
  │  · 过滤页眉、版权信息、DOI 等垃圾段落
  │
  ▼
[retriever.py] 两阶段检索
  │  · Embedding: all-MiniLM-L6-v2（向量召回）
  │  · Reranker: BAAI/bge-reranker-base（Cross-Encoder 精排）
  │  · 分层配额召回：document 全取 + section 全取 + paragraph top-K
  │  · diverse_top_k：保证三层都有代表进入最终结果
  │
  ▼
[cli.py] Function Calling Agent
     · DeepSeek API + tool use
     · 自动决定是否需要检索、检索什么关键词
     · 支持单轮和多轮检索
     · 基于检索结果生成有引用来源的回答
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/paper-assistant.git
cd paper-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 4. 放入论文

将 PDF 论文放到 `data/papers/` 目录下。

### 5. 运行

```bash
python cli.py
```

首次运行会下载 Embedding 和 Reranker 模型（约 1.3 GB），之后会自动缓存。

> 国内用户如遇 HuggingFace 下载超时，可设置环境变量：
> ```bash
> set HF_ENDPOINT=https://hf-mirror.com   # Windows
> export HF_ENDPOINT=https://hf-mirror.com # Linux/Mac
> ```

## 项目结构

```
paper_assistant/
├── cli.py                 # 主程序入口（Agent + 交互循环）
├── src/
│   ├── paper_parser.py    # PDF 解析与结构识别
│   ├── chunker.py         # 层次化 chunk 构建
│   └── retriever.py       # Embedding + Reranker 检索
├── data/
│   └── papers/            # 放置 PDF 论文
├── requirements.txt
├── .env.example           # API Key 配置模板
└── .gitignore
```

## 设计决策

### 为什么用层次化 Chunking？

论文助手需要同时回答两类问题：

| 问题类型 | 示例 | 需要的粒度 |
|---------|------|-----------|
| 宏观理解 | "这篇论文讲了什么？" | 文档级（Abstract） |
| 方法概述 | "实验怎么做的？" | 章节级（Experiments） |
| 细节查询 | "字典大小是多少？" | 段落级（具体参数） |

单一粒度的 chunking 无法同时满足两类需求。层次化索引让每种查询都能找到最合适的抽象层级。

### 为什么需要分层召回？

在统一索引中，段落级 chunk 数量远多于文档级和章节级（约 100:6:1）。如果用纯分数排序召回，段落级会凭数量优势挤占所有名额，导致文档级和章节级永远无法被 Reranker 看到。

分层配额召回保证每个层级都有代表进入重排阶段——本质上是解决数据分布不平衡问题，类似于机器学习中的类别加权策略。

### 为什么用 diverse_top_k 而不是纯分数排序？

Reranker 擅长"精确匹配"但不擅长"抽象匹配"。对于"论文的核心贡献是什么"这类总结性问题，Abstract 的 Reranker 分数可能很低（因为 Abstract 不会直接写"core contribution"），但它恰恰是 LLM 最需要的上下文。

diverse_top_k 确保不同层级的最佳 chunk 都进入最终结果，让 LLM 获得多角度的上下文来生成高质量答案。

## 已知局限

- 目前仅支持单篇论文（多论文支持在计划中）
- PDF 表格和数学公式的提取质量有限
- 章节识别基于正则匹配，对非标准格式论文可能失败
- Embedding 模型（all-MiniLM-L6-v2）为英文模型，对中文论文效果较差

## 技术栈

- **LLM**: DeepSeek Chat API（兼容 OpenAI SDK）
- **Embedding**: all-MiniLM-L6-v2（sentence-transformers）
- **Reranker**: BAAI/bge-reranker-base（Cross-Encoder）
- **PDF 解析**: PyMuPDF
- **Agent**: 原生 Function Calling（无框架依赖）