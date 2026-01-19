# 阻塞式输出 invoke() 不再赘述

# 流式输出 stream()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

s_m = SystemMessage(content="你是一个边缘计算专家,精通边缘计算领域的知识")
h_m = HumanMessage(content="你擅长什么?")
a_m = AIMessage(content="边缘计算是...")
# c_m = ChatMessage(role="", content="补充...")

messages = [s_m, h_m]
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
