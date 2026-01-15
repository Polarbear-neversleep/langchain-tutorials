# ConversationBufferWinodowMemory 记住前k条信息
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
import os
import dotenv

memory = ConversationBufferWindowMemory(k=2, return_messages=True)
# return_messages确定返回str还是message
memory.save_context({"input": "你好"}, {"output": "怎么了"})
memory.save_context({"input": "你是谁"}, {"output": "我是人工智能助手小智"})
memory.save_context({"input": "请你介绍一下自己"}, {"output": "当然了,我是无所不能的小智"})
print(memory.load_memory_variables({}))

# ConversationTokenBufferMemory 通过限制token来限制memory长度
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=20)  # 不同llm的token不一样
memory.save_context({"input": "你好吗？"}, {"output": "我很好"})
memory.save_context({"input": "今天天气怎么样？"}, {"output": "晴天30度"})

print(memory.load_memory_variables({}))
