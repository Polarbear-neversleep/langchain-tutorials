from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent,AgentExecutor

import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# 提示词模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是乐于助人的AI助手,根据用户问题进行回答,必要时可以使用互联网搜索工具Search"),
        ("human","{input}"),
        ("placeholder","{agent_scratchpad}")
    ]
)

# 工具调用
search = TavilySearch()
search_tool = Tool(
    name = "Search",
    func=search.run,
    description= "用于检索互联网上的信息"
)

# 大模型
llm = ChatOpenAI(model = "gpt-4o-mini")

# Agent实例 使用Function_call
agent = create_tool_calling_agent(
    llm = llm,
    tools=[search_tool],
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent = agent,
    tools=[search_tool],
    verbose=True
)
response = agent_executor.invoke({"input":"广州明天的天气如何?"})
print(response)