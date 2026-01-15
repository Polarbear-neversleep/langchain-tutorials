from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool,StructuredTool
import os
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults # 搜索工具
from langchain.memory import ConversationBufferMemory
import uuid 

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVLIY_API_KEY"] = os.getenv("TAVILY_API_KEY")

search = TavilySearchResults(max_results = 3)
# 自定义工具
# search_tool = StructuredTool.from_function(
#     func=search.run,
#     name="Search",
#     description="用于检索互联网上的信息"
# )
search_tool = Tool(
    func=search.run,
    name="Search",
    description="用于检索互联网上的信息"
) 
llm = ChatOpenAI(model="gpt-4o-mini")

# 嵌入记忆模块 ConversationBufferMemory
memory = ConversationBufferMemory(
    return_messages= True,
    memory_key="chat_history"
)

# 获取Agent实例
# 使用ReACT模型 有记忆版本
# agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION
# 使用FunctionCall模型
agent = AgentType.OPENAI_FUNCTIONS
agent_executor = initialize_agent(
    tools = [search_tool], #可以直接用search，但无法定义description
    llm = llm,
    agent = agent,
    memory = memory,
    verbose = True
)

response = agent_executor.invoke("今日广州番禺区的天气如何？")
print(response)
response_2 = agent_executor.invoke("明天呢？")
print(response_2)
