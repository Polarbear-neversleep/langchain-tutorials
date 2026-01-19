# System Human Chat AI Message

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

s_m = SystemMessage(content="你是一个边缘计算专家")
h_m = HumanMessage(content="请你简单介绍边缘计算的研究领域有哪些?")
a_m = AIMessage(content="边缘计算是...")
c_m = ChatMessage(role="", content="补充...")

messages = [s_m, h_m]
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke(messages)
print(response.content)
