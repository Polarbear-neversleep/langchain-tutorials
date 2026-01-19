# ConversationChain 对ConversationBufferMemory和LLMChain进行封装，并提供了默认提示词模板
from langchain.chains.conversation.base import ConversationChain
import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(model="gpt-4o-mini")

chain = ConversationChain(llm=llm)
response_1 = chain.invoke({"input", "你好,我的名字叫小明"})
response_2 = chain.invoke({"input", "你知道我是谁吗？"})
print(response_2)
