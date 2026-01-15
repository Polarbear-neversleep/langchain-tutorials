# 在基类ChatMessageHistory的子类ConversationBufferMemory(存储完整对话历史)
from langchain.memory import ConversationBufferMemory
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(model="gpt-4o-mini")
history = ConversationBufferMemory()

# 保存消息
history.save_context(
    inputs={"input": "你好,我是小明"},
    outputs={"output": "你好小明,我是人工智能助手小智"},
)
history.save_context(inputs={"human": "请问1~10之和是多少"}, outputs={"ai": "55"})

# 输出消息
# print(history.load_memory_variables({}))
# print(history.chat_memory.messages)

# 使用PromptTemplate模板
prompt_template = PromptTemplate.from_template(
    template="""
    你可以与人类对话.
    当前对话历史: {history}
    人类问题: {question}
    """
)  # 不包含当前对话内容
chain = LLMChain(llm=llm, prompt=prompt_template, memory=history)
response = chain.invoke({"question:", "我刚刚问了什么?"})
print(history)
print(response)

# 使用ChatPromptTemplate模板
memory = ConversationBufferMemory(return_messages=True)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个与人类对话的机器人"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "问题：{question}")
    ]
)  # 包含当前对话内容
chain = LLMChain(prompt=prompt_template, llm=llm, memory=memory)
chain.invoke({"question": "中国首都在哪里?"})
response = chain.invoke({"question": "我刚刚问了什么？"})
print(response)
print(memory.load_memory_variables({}))
