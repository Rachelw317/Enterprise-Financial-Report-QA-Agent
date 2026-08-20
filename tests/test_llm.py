from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv

load_dotenv()

def get_deepseek():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到API_KEY。")

    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.2,
        api_key=api_key,
    )
    return model


query = "聚水潭公司2025年的收入是多少亿元？"

