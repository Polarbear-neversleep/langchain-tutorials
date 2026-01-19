from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import create_react_agent,AgentExecutor
from langchain.memory import ConversationBufferMemory
import langchain.hub as hub
import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# 提示词模板
prompt_template = hub.pull("hwchase17/react-chat")

# 工具调用
search = TavilySearch()
search_tool = Tool(
    name = "Search",
    func=search.run,
    description= "用于检索互联网上的信息"
)

# 大模型
llm = ChatOpenAI(model = "gpt-4o-mini")

# 嵌入记忆模块
memory = ConversationBufferMemory(
    return_messages= True,
    memory_key="chat_history"
)

# Agent实例 使用ReACT 注意不是这里用memory
agent = create_react_agent(
    llm = llm,
    tools=[search_tool],
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent = agent,
    tools=[search_tool],
    verbose=True,
    memory=memory,
    handle_parsing_errors=True #提升Agent容错能力
)
response = agent_executor.invoke({"input":"广州明天的天气如何?"})
print(response)
response = agent_executor.invoke({"input":"上海的呢?"})
print(response)