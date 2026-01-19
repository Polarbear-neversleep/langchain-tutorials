# Ollama本地大模型
# langchain调用本地大模型
from langchain_ollama import ChatOllama
import time

st = time.time()
llm = ChatOllama(model="gemma3:1b",disable_streaming=False)
# response = llm.invoke("请你编写快速排序的代码")
for chunk in llm.stream("请你编写快速排序的代码"):
    print(chunk.content,end="",flush=True)
ft = time.time()
print(ft-st)
# print(response.content)
