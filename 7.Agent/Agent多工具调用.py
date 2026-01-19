from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool,StructuredTool
import os
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults # 搜索工具
from langchain_experimental.tools import PythonREPLTool #数学工具

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
python_repl = PythonREPLTool()
calc_tool = Tool(
    name="Caculator",
    func=python_repl.run,
    description="用于执行数学计算,例如计算百分比变化"
)
llm = ChatOpenAI(model="gpt-4o-mini")

# 获取Agent实例
agent_executor = initialize_agent(
    tools = [search_tool,python_repl], #可以直接用search，但无法定义description
    llm = llm,
    agent = AgentType.OPENAI_FUNCTIONS,  #Funtion call使用 AgentType.OPENAI_FUNCTIONS
    verbose = True
)

response = agent_executor.invoke("2024年广州总降水量是多少?相比2023年降水量上升或下降了多少个百分点?")
print(response)
