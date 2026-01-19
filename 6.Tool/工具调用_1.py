# 自定义实现工具调用
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool

# 方法一@tool
@tool(name_or_callable="add_two_number", description="add two numbers", return_direct=True)
def add_number(a: int, b: int) -> int:
    """ 计算两个整数的和 """
    return a+b


print("name=", add_number.name)
print("args=", add_number.args)
print("description=", add_number.description)
print("return_direct", add_number.return_direct)

# 方法二 StructuredTool
def search_google(query:str):
    return f"您想要查询的是:{query}"

search = StructuredTool.from_function(
    func=search_google,
    name="search",
    description="使用google查询相关信息"
)
print(search)
print("name=", search.name)
print("args=", search.args)
print("description=", search.description)

# 工具调用时response.additional_kwargs有function_call，同时content为空，若不调用则content不为空