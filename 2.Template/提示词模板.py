# PromptTemplate 生成字符串提示
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.schema import HumanMessage

prompttemplate = PromptTemplate(
    template="你是{name},是{field}领域的专家", input_variables=["name", "field"])
# 推荐 from_template
prompttemplate2 = PromptTemplate.from_template(
    "你是{name},是{field}领域的专家", partial_variables={"name": "小花"})

# partial_format 或者 partial 来给部分变量赋值

# format格式 返回str
prompt = prompttemplate2.format(name="小智", field="人工智能")
# invoke格式 返回promptvalue
prompt = prompttemplate2.invoke(input={"name": "小智", "field": "人工智能"})
print(prompt)

# ChatPromptTemplate
chatprompttemplate = ChatPromptTemplate(messages=[("system", "你是一个AI助手,你的名字是{name}"), ("human", "我的问题是{question}")], input_variables=["name", "question"]) #input_variables 可要可不要
chatprompttemplate2 = ChatPromptTemplate.from_messages([("system", "你是一个AI助手,你的名字是{name}"), ("human", "我的问题是{question}")])
# invoke格式 返回promptvalue
prompt = chatprompttemplate2.invoke(input={"name": "小智", "question": "1+1=?"})
print(prompt)
# format格式 返回str
prompt2 = chatprompttemplate2.format(name="小智", question="1+1=?") 
print(prompt2)
# from_messages格式 返回列表
prompt3 = chatprompttemplate2.format_messages(name="小智", question="1+1=?") 
print(prompt3)
# from_prompt格式 返回promptvalue
prompt4 = chatprompttemplate2.format_prompt(name="小智", question="1+1=?")
print(prompt4)
# promptvalue转str
print(prompt4.to_string())

# MessagePlaceHolder:不确定Message个数,存储历史对话
chatprompttemplate3 = ChatPromptTemplate.from_messages([("system", "你是一个AI助手,你的名字是{name}"), MessagesPlaceholder("msgs")])
prompt5 = chatprompttemplate3.invoke(input={"name": "小智", "msgs": [HumanMessage("我的问题是: 等差数列的求和公式?")]}) # type: ignore
print(prompt5)

# 调用大模型
from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model="gpt-4o-mini")

# respose = llm.invoke(prompt5)
# print(respose.content)


# FewShotPromptTemplate 提供少量样本引导大模型输出
example_prompt = PromptTemplate.from_template(template="{input}\noutput:{output}")
examples = [
    {"input": "1+1=2", "output": "数学"},
    {"input": "唐诗宋词元曲", "output": "文学"},
    {"input": "牛顿第二定律", "output": "物理"},
    {"input": "氢氦锂铍硼", "output": "化学"}
]

few_shot_template = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="{input}\noutput:",
    input_variables=["input"]
)

few_shot_prompt = few_shot_template.invoke({"input": "质壁分离"})
# few_shot_response = llm.invoke(few_shot_prompt)
# print(few_shot_response.content)

# FewShotChatMessagePromptTemplate 加入Message(system/human)
examples = [
    {"input": "1+1=2", "output": "数学"},
    {"input": "唐诗宋词元曲", "output": "文学"},
    {"input": "牛顿第二定律", "output": "物理"},
    {"input": "氢氦锂铍硼", "output": "化学"}
]
example_prompt = ChatPromptTemplate.from_messages([("human", "以下内容涉及什么学科:{input}"), ("ai", "{output}")])
few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你非常熟悉中学的课程内容"),
        few_shot_chat_prompt,
        ("human", "{input}")
    ]
)
final_response = llm.invoke(final_prompt.invoke({"input": "质壁分离"}))
print(final_response.content)
