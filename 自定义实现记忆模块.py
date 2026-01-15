# 自定义实现memory模块 存储过去输出和输入
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model="gpt-4o-mini")


def chat_with_model(answer):

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个人工智能助手"),
            ("human", "{question}")
        ]
    )

    while True:
        chain = prompt_template | llm
        response = chain.invoke({"question": answer})
        print(f"模型回复:{response.content}")

        user_input = input("你还有其他问题吗？(输入'退出'时结束会话)")

        prompt_template.messages.append(AIMessage(content=response.content))
        prompt_template.messages.append(HumanMessage(content=user_input))
        if user_input == '退出':
            print("结束会话！")
            break


chat_with_model("你好,很高兴认识你!")
