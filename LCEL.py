# LangChain Expression Language
# 通道| LLMChain
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# LLMChain
llm_api = ChatOpenAI(model="gpt-4o-mini") 
llm_loc = ChatOllama(model="deepseek-r1:7b")

# 使用prompttemplate
prompt_template = PromptTemplate.from_template(template="回答用户以下问题:{question}")
chain = LLMChain(
    llm=llm_api,
    prompt=prompt_template,
    verbose=True
)
# response = chain.invoke(input={"question": "中秋节是哪个国家的节日?"})
# print(response) # 返回字典

# 使用chatprompttemplate
chatprompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个了解{field}的专家"),
        ("human", "请你根据实时依据解释以下问题{question}")
    ]
)
chain2 = LLMChain(
    llm=llm_loc,
    prompt=chatprompt_template,
    verbose=True
)
response = chain.invoke(input={"field": "客观事实", "question": "中秋节是哪个国家的节日"})
print(response)
