# 大模型本身没有上下文记忆能力 ,需要合并到单次问答才能成功回答
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

s1 = SystemMessage("我是人工智能助手小智")
h1 = HumanMessage("什么是边缘计算?")
s2 = SystemMessage("我可以教你很多生活小技巧")
h2 = HumanMessage("你叫什么名字?")

messages = [s1, h1, s2, h2]

llm = ChatOpenAI(model="gpt-4o-mini")
res = llm.invoke(messages)
print(res.content)
