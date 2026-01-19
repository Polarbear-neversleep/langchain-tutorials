from langchain_openai import ChatOpenAI

# 硬编码方式
# llm = ChatOpenAI(
#     model="gpt-4o-mini",  # 默认是gpt3.5-turbo
#     base_url="https://api.openai-proxy.org/v1",
#     api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
# )
# str = llm.invoke("写一首诗")
# print(str)
# print(type(str))

# 环境变量方式 Pycharm实现
# import os

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY1")
# os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")


# 配置文件方式 使用.env文件
# 方式一:
# import os
# import dotenv
# dotenv.load_dotenv()
# # print(os.getenv("OPENAI_BASE_URI"))
# # print(os.getenv("OPENAI_API_KEY"))
# llm = ChatOpenAI(
#     model="gpt-4o-mini",  # 默认是gpt3.5-turbo
#     base_url=os.getenv("OPENAI_BASE_URI"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     max_tokens=20,
#     temperature=0.7
# )
# response = llm.invoke("你能理解到最新的数据是什么时间?")
# print(response.content)


# 方式二
import os
import dotenv
import time
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
st_time = time.time()
llm = ChatOpenAI(
    model="gemini-2.5-flash-lite-preview-09-2025",  # 默认是gpt3.5-turbo
)
# respose = llm.invoke("边缘计算中如果某个节点出现单点故障,应该如何处理和善后?请分步回答。")
response = llm.invoke("人类脱发的原因有哪些？")
ed_time = time.time()
print(ed_time - st_time)
print(type(response))
print(response)
print(len(response.json()["response"].split()))
