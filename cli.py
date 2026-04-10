import os
from dotenv import load_dotenv
from openai import OpenAI
from src.load_all_paper import load_all_papers
from src.agent import SYSTEM_PROMPT, run_agent_turn

load_dotenv()

CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

if __name__ == "__main__":
    print("=" * 50)
    print("Paper Assistant - 论文智能问答助手")
    print("=" * 50)

    CHUNKS, VECTORS = load_all_papers()

    if not CHUNKS:
        print("没有找到论文，请将 PDF 文件放入 data/papers/ 目录")
        exit(1)

    paper_ids = set(c["paper_id"] for c in CHUNKS)
    print(f"\n已加载 {len(paper_ids)} 篇论文，共 {len(CHUNKS)} 个 chunks")
    print("\n已加载的论文：")
    for pid in sorted(paper_ids):
        count = sum(1 for c in CHUNKS if c["paper_id"] == pid)
        print(f"  · {pid} ({count} chunks)")

    print()
    print("你可以开始提问了（输入 'exit' 退出，'new' 开启新对话）")
    print("=" * 50)

    # 多轮对话的 messages 历史
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("再见！")
            break
        if user_input.lower() == 'new':
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("已开启新对话。")
            continue

        messages.append({"role": "user", "content": user_input})
        answer, messages = run_agent_turn(messages, CHUNKS, VECTORS, CLIENT)
        print(f"\n助手: {answer}")
