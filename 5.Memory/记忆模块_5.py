# ConversationSummaryMemory 对过去的对话做摘要
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory, ConversationSummaryBufferMemory

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(model="gpt-4o-mini")
history = ChatMessageHistory()
history.add_user_message("你好,我是小明")
history.add_ai_message("你好小明,很高兴认识你,我是人工智能助手小智!")
memory = ConversationSummaryMemory(llm=llm, chat_memory=history)
print(memory)
# 加入新的对话内容
memory.save_context(inputs={"human": "1~10之和是多少"}, outputs={"ai": "55"})
print(memory.load_memory_variables({}))

# ConversationSummaryBufferMemory结合了Buffer整段存储的特点和Summary摘要的特点
# 通过设置token来区分较早信息还是较新信息
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40, return_messages=True)
memory.save_context(inputs={"input": "你好,我是小明"}, outputs={"output": "你好小明,很高兴认识你,我是人工智能助手小智!"})
memory.save_context(inputs={"input": "今天天气如何"}, outputs={"output": "晴天25度, 天气不错"})
memory.save_context(inputs={"input": "1~10之和是多少"}, outputs={"output": "55"})

print(memory.load_memory_variables({}))
