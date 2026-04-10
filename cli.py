# cli.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.paper_parser import parse_paper
from src.chunker import build_hierarchical_chunks
from src.retriever import build_index, search_with_rerank
from src.load_all_paper import load_all_papers
from src.tools import search_paper, get_paper_abstract, compare_papers,list_papers
load_dotenv()

CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ====== 全局变量（加载论文后填充）======
global CHUNKS
global VECTORS

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
SYSTEM_PROMPT = """你是一个学术论文阅读助手。你可以帮助用户理解和分析已加载的多篇论文。

你可以使用 search_paper 工具在所有已加载的论文中检索相关内容。检索结果会标注来自哪篇论文、哪个章节。

回答问题时请遵循以下原则：
1. 基于论文内容回答，不要编造论文中没有的信息
2. 在回答中明确指出信息来自哪篇论文的哪个章节
3. 如果多篇论文涉及同一话题，对比它们的异同
4. 如果检索结果不足以回答问题，诚实地说"已加载的论文中没有明确提到这一点"
5. 用中文回答用户的问题
"""

# cli.py 里的 execute_tool
def execute_tool(tool_name, tool_args):
    if tool_name == "list_papers":
        return list_papers(CHUNKS)
    elif tool_name == "get_paper_abstract":
        return get_paper_abstract(tool_args["paper_name"], CHUNKS)
    elif tool_name == "compare_papers":
        return compare_papers(tool_args["paper_a"], tool_args["paper_b"],
                              tool_args.get("aspect", ""), CHUNKS)
    elif tool_name == "search_paper":
        return search_paper(tool_args["query"], CHUNKS, VECTORS)
    else:
        return f"未知工具: {tool_name}"

# ====== Agent 主循环 ======
def run_agent(user_question: str, max_steps: int = 10) -> str:
    """
    这就是你之前 agent_v2.py 里的 run_agent，
    只是工具从 calculator/get_weather/search 换成了 search_paper。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    
    for step in range(max_steps):
        # 1. 调用 LLM
        response = CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        
        msg = response.choices[0].message
        
        # 2. 如果没有工具调用，返回最终答案
        if not msg.tool_calls:
            print(f"\n助手: {msg.content}")
            return msg.content
        
        # 3. 把 assistant 消息追加回去
        messages.append(msg.model_dump())
        
        # 4. 执行所有工具调用
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"  [调用工具] {tool_name}({tool_args})")
            
            # 执行工具
            tool_result = execute_tool(tool_name, tool_args)
            
            # 追加工具结果
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call.id,
            })
        
    print("达到最大步数，停止。")
    return "抱歉，处理超时。"

# ====== 主程序 ======
if __name__ == "__main__":
    print("=" * 50)
    print("📄 Paper Assistant - 论文智能问答助手")
    print("=" * 50)
    
    # 加载所有论文

    CHUNKS, VECTORS = load_all_papers()
    
    if not CHUNKS:
        print("没有找到论文，请将 PDF 文件放入 data/papers/ 目录")
        exit(1)
    
    # 统计论文信息
    paper_ids = set(c["paper_id"] for c in CHUNKS)
    print(f"\n已加载 {len(paper_ids)} 篇论文，共 {len(CHUNKS)} 个 chunks")
    print("\n已加载的论文：")
    for pid in sorted(paper_ids):
        count = sum(1 for c in CHUNKS if c["paper_id"] == pid)
        print(f"  · {pid} ({count} chunks)")
    
    print()
    print("你可以开始提问了（输入 'exit' 退出）")
    print("=" * 50)
    
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("再见！")
            break
        run_agent(user_input)