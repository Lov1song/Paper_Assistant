# cli.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.paper_parser import parse_paper
from src.chunker import build_hierarchical_chunks
from src.retriever import build_index, search_with_rerank

load_dotenv()

CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ====== 全局变量（加载论文后填充）======
CHUNKS = []
VECTORS = None

# ====== 工具函数 ======
def search_paper(query: str) -> str:
    """在论文中检索相关内容"""
    results = search_with_rerank(query, CHUNKS, VECTORS, recall_k=20, final_k=5)
    
    if not results:
        return "未找到相关内容。"
    
    formatted_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        section = chunk.get("section_title", "未知章节")
        level = chunk["level"]
        text = chunk["text"]
        formatted_parts.append(
            f"[结果 {i}] (来源: {section} | 类型: {level} | 相关度: {score:.3f})\n{text}"
        )
    
    return "\n\n---\n\n".join(formatted_parts)

# ====== 工具 Schema ======
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "在已加载的论文中检索与问题相关的内容。返回论文中最相关的段落，标注来源章节。当用户提出任何关于论文内容的问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词，用英文，尽量具体。例如 'sparse dictionary method' 或 'evaluation dataset and metrics'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ====== System Prompt ======
SYSTEM_PROMPT = """你是一个学术论文阅读助手。你的任务是帮助用户理解和分析已加载的论文。

你可以使用 search_paper 工具在论文中检索相关内容。检索结果会标注来源章节和内容类型。

回答问题时请遵循以下原则：
1. 基于论文内容回答，不要编造论文中没有的信息
2. 在回答中引用来源章节，例如"根据论文的 Section 3..."
3. 如果检索结果不足以回答问题，诚实地说"论文中没有明确提到这一点"
4. 对于概括性问题，优先参考 Abstract 和 Introduction 的内容进行总结
5. 用中文回答用户的问题
"""

# ====== Agent 主循环 ======
def run_agent(user_question: str, max_steps: int = 5) -> str:
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
            
            print(f"  [检索中] query: {tool_args.get('query', '')}")
            
            # 执行工具
            if tool_name == "search_paper":
                tool_result = search_paper(tool_args["query"])
            else:
                tool_result = f"未知工具: {tool_name}"
            
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
    # 1. 加载论文
    pdf_path = "data/papers/1.2016-Arad_and_Ben_Shahar-Sparse_Recovery_of_Hyperspectral_Signal_from_Natural_RGB_Images.pdf"
    
    print(f"正在加载论文: {pdf_path}")
    paper = parse_paper(pdf_path)
    
    print("正在构建索引...")
    CHUNKS = build_hierarchical_chunks(paper, paper_id="arad_2016")
    VECTORS = build_index(CHUNKS)
    
    print(f"\n加载完成！")
    print(f"  论文: {paper['metadata']['filename']}")
    print(f"  页数: {paper['metadata']['num_pages']}")
    print(f"  章节: {len(paper['sections'])} 个")
    print(f"  索引: {len(CHUNKS)} 个 chunks")
    print()
    print("=" * 50)
    print("论文助手已就绪，你可以开始提问了")
    print("输入 'exit' 退出")
    print("=" * 50)
    
    # 2. 交互循环
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("再见！")
            break
        
        run_agent(user_input)