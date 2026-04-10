import uuid
import requests

BASE_URL = "http://127.0.0.1:8000"
def ask(session_id: str, question: str) -> str:
    """向 /chat 发送请求，返回 answer 字符串"""
    #发送请求
    response = requests.post(f"{BASE_URL}/chat", json={
        "session_id": session_id,
        "question": question,
    })
    #解析响应
    if response.status_code == 200:
        data = response.json()
        return data["answer"]
    else:
        print(f"请求失败: {response.status_code} - {response.text}")
        return "抱歉，发生了错误。"



def new_session() -> str:
    """生成一个新的 session_id"""
    return str(uuid.uuid4())


if __name__ == "__main__":
    session_id = new_session()
    print(f"会话已开始（session: {session_id[:8]}...）")
    print("输入 'new' 开启新对话，'exit' 退出")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            #退出逻辑
            print("再见！")
            break

        if user_input.lower() == "new":
            #重置 session_id，打印提示
            requests.delete(f"{BASE_URL}/session/{session_id}")  # 通知服务端清除                                              
            session_id = new_session()
            print(f"已开启新对话（session: {session_id[:8]}...）")
            continue
        

        #调用 ask()，打印结果
        answer = ask(session_id,user_input)
        print(f"\n助手: {answer}")