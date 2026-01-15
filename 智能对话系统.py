# 智能对话助手
# 功能：调用语言模型，创建检索器RAG，使用搜索工具，保留聊天历史(不同id对话历史不同,字典实现)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.agents import AgentExecutor,create_tool_calling_agent,create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import Tool
import langchain.hub as hub

import dotenv
import os

dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 提示词模板 template or message
prompt_template = PromptTemplate.from_template("你是智能机器人助手小明,你需要帮助用户解决问题") #system prompt
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个AI助手{name},你帮助用户解决问题"),
        ("human","我的问题是{question}")
    ]
)
prompt_fc = hub.pull("hwchase17/openai-functions-agent")
prompt_react = hub.pull("hwchase17/react")
# 构造RAG
# 文件加载
text_loader = TextLoader(file_path="./李白.txt",encoding="utf-8")
docs = text_loader.load()

# 数据切分 循环切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 60,
    chunk_overlap = 10,
    add_start_index = True
)
docs = text_splitter.split_documents(docs)
# 文本嵌入
embedding_model = OpenAIEmbeddings(model = "text-embedding-ada-002") #不嵌入了，直接加入数据库
# 数据存储
db = Chroma.from_documents(
    embedding = embedding_model,
    documents=docs,
    persist_directory="./chroma_libai"
) 
# 数据检索
retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs = {"fetcg":3}
)

# 大模型调用
llm = ChatOpenAI(model = "gpt-4o-mini")

# Agent构建
# 构造工具集
# 搜索工具
search = TavilySearchResults(max_results = 3)
search_tool = Tool(
    name = "Search",
    func = search.run,
    description="用于检索互联网上的信息"
)
# 检索工具
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="libai_search",
    description="搜索关于李白的信息"
)
tools = [search_tool,retriever_tool]

# 历史对话保存
history = ConversationBufferWindowMemory(memory_key="chat_history")

# ReAct
react_agent = create_react_agent(
    llm = llm,
    tools= tools,
    prompt = prompt_react
)
# FuctionCall
fc_agent = create_tool_calling_agent(
    llm=llm,
    tools = tools,
    prompt=prompt_fc
)

# 开启智能对话
sessions = dict()
while True:
    id = input("请输入你的聊天编号:")
    content = input(f"你好{id},请输入聊天内容:")
    if not sessions.get(id):
        agent_type = input("你是新用户,回答你的问题前,请你选择工具调用的模式:(ReACT or FunctionCall)")
        if agent_type == "ReACT":
            react_executor = AgentExecutor(
                agent = react_agent,
                tools = tools,
                memory = history,
                verbose = True
            )
            # agent_with_chat_history = RunnableWithMessageHistory(
            #     runnable = react_executor,
            #     get_session_history=history,
            #     input_messages_key = "input",
            #     history_messages_key = "chat_history"
            # )
            sessions[id] = react_executor
        else:
            fc_executor = AgentExecutor(
                agent = react_agent,
                tools = tools,
                memory = history,
                verbose = True
            )
            # agent_with_chat_history = RunnableWithMessageHistory(
            #     runnable = fc_executor,
            #     get_session_history= history,
            #     input_messages_key = "input",
            #     history_messages_key = "chat_history"
            # )
            sessions[id] = fc_executor
    agent = sessions[id]
    
    response = agent.invoke({"input":content})
    # 加入对话历史
    
    print(response)