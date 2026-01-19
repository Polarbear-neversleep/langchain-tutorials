# 记忆模块的API ChatMessageHistory 仅作为保留消息的容器(内部结构，防止暴露不建议使用)
from langchain.memory import ChatMessageHistory
import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model="gpt-4o-mini")

history = ChatMessageHistory()
history.add_user_message("你好")
history.add_ai_message("你好,我是人工智能机器人小智")
history.add_user_message("你叫什么名字？")

print(history.messages)
response = llm.invoke(history.messages)
print(response)
