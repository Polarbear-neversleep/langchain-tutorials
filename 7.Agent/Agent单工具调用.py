from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool,StructuredTool
import os
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults # 搜索工具
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

# 获取Agent实例
# 使用ReACT模型
agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
# 使用FunctionCall模型
agent = AgentType.OPENAI_FUNCTIONS
agent_executor = initialize_agent(
    tools = [search_tool], #可以直接用search，但无法定义description
    llm = llm,
    agent = agent,
    verbose = True
)

# response = agent_executor.invoke("今日广州番禺区的天气如何？")
# print(response)


def calculator(expression:str):
    print("调用计算器")
    return str(eval(expression))
calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description="计算字符串构成的表达式,例如12*3=36, 1+2*3=7等等"
)
agent_executor_2 = initialize_agent(
    llm=llm,
    tools=[calc_tool],
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
response = agent_executor_2.invoke("计算一下式子:2**3+6**2+8/2+3*2")
print(response)