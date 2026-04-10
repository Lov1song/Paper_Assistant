import os
import json
from openai import OpenAI
from src.tools import search_paper, get_paper_abstract, compare_papers, list_papers

# ====== 工具 Schema ======
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "在所有已加载的论文中检索与问题相关的内容。返回最相关的段落，标注来自哪篇论文和哪个章节。当用户提出任何关于论文内容的问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词，用英文，尽量具体"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "列出所有已加载的论文清单，包括论文名称和索引信息。当用户询问'有哪些论文'、'加载了几篇'、'库里有什么'时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_abstract",
            "description": "获取指定论文的摘要内容。当用户询问某篇特定论文讲了什么、想了解某篇论文的概要时使用。比 search_paper 更快更精准，适合概括性问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_name": {
                        "type": "string",
                        "description": "论文名称或关键词，例如 'Mamba' 或 'MST' 或 'sparse recovery'"
                    }
                },
                "required": ["paper_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": "对比两篇论文的内容。返回两篇论文的摘要和相关信息，便于分析异同。当用户要求对比、比较两篇论文时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_a": {
                        "type": "string",
                        "description": "第一篇论文的名称或关键词"
                    },
                    "paper_b": {
                        "type": "string",
                        "description": "第二篇论文的名称或关键词"
                    },
                    "aspect": {
                        "type": "string",
                        "description": "对比的维度，例如 'method'、'performance'、'dataset'。留空则返回整体概要。"
                    }
                },
                "required": ["paper_a", "paper_b"]
            }
        }
    }
]

# ====== System Prompt ======
SYSTEM_PROMPT = """你是一个学术论文阅读助手，帮助用户理解和分析已加载的多篇学术论文。

## 可用工具
你有以下工具：
- search_paper：在所有论文中语义检索相关段落。适合查找具体细节、方法描述、实验数据。
- list_papers：列出所有已加载论文。适合用户问"有哪些论文"或你需要确认论文名称时。
- get_paper_abstract：获取指定论文的摘要。适合用户问"XX论文讲了什么"这类概括性问题。
- compare_papers：获取两篇论文的摘要用于对比。适合用户要求对比两篇论文时。

## 工具使用策略
1. 概括性问题（"讲了什么/核心贡献"） → 优先用 get_paper_abstract
2. 细节问题（"用了什么方法/数据集/指标"） → 用 search_paper，query 用英文
3. 对比问题（"A和B有什么区别"） → 先用 compare_papers，再用 search_paper 补充细节
4. 列表问题（"有哪些论文/哪些用了X"） → 先用 list_papers 确认范围，再用 search_paper 逐个确认
5. 如果第一次检索结果不够，换不同的关键词再搜一次，但不要超过3次检索

## 回答原则
1. 先给出简洁的直接回答（1-2句话），再展开详细解释
2. 必须基于论文内容，不要编造论文中没有的信息
3. 引用来源时注明论文名称和章节
4. 涉及多篇论文时，用结构化的方式（如分点或表格）组织对比
5. 检索结果不足时，诚实说明"已加载的论文中未找到相关信息"
6. 用中文回答，但检索 query 使用英文（因为论文是英文的）
"""


def execute_tool(tool_name: str, tool_args: dict, chunks: list, vectors) -> str:
    if tool_name == "list_papers":
        return list_papers(chunks)
    elif tool_name == "get_paper_abstract":
        return get_paper_abstract(tool_args["paper_name"], chunks)
    elif tool_name == "compare_papers":
        return compare_papers(tool_args["paper_a"], tool_args["paper_b"],
                              tool_args.get("aspect", ""), chunks)
    elif tool_name == "search_paper":
        return search_paper(tool_args["query"], chunks, vectors)
    else:
        return f"未知工具: {tool_name}"


def run_agent_turn(messages: list[dict], chunks: list, vectors, client: OpenAI,
                   max_steps: int = 10) -> tuple[str, list[dict]]:
    """
    执行一轮 Agent 循环。
    接收完整的 messages 历史，追加执行结果后返回。

    Returns:
        (answer, updated_messages)
    """
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, messages

        messages.append(msg.model_dump())

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  [调用工具] {tool_name}({tool_args})")

            tool_result = execute_tool(tool_name, tool_args, chunks, vectors)

            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call.id,
            })

    return "抱歉，处理超时。", messages
