from langchain_openai import ChatOpenAI
import dotenv
import os
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    XMLOutputParser,
    CommaSeparatedListOutputParser
)
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate

# LangChain顺序 prompt -> llm -> parser

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(model="gpt-4o-mini")
# response = llm.invoke("大模型是什么")

# 使用StrOutputParser输出
str_parser = StrOutputParser()
# str_response = str_parser.invoke(response)
# print(str_response)

# 使用JsonOutputParser输出
js_parser = JsonOutputParser()
prompt_template = PromptTemplate.from_template(
    template="回答用户问题\n按照以下格式{instructions}\n问题为{question}\n",
    partial_variables={"instructions": js_parser.get_format_instructions()},
)
prompt = prompt_template.invoke({"question": "台风如何形成?"})
# response = llm.invoke(prompt)
# json_response = js_parser.invoke(response)
# print(json_response)
# ***拓展：使用管道符
chain = prompt_template | llm | js_parser
# json_result = chain.invoke({"question": "飓风如何形成?"})
# print(json_result)

# 使用XMLOutputParser输出
xml_parser = XMLOutputParser()
prompt_template = PromptTemplate.from_template(
    template="回答用户问题\n按照以下格式{instructions}\n问题为{question}\n"
)
prompt_template2 = prompt_template.partial(
    instructions=xml_parser.get_format_instructions()
)
# response = llm.invoke(prompt_template2.invoke({"question": "生成成龙电影记录简介"}))
# print(response.content)
# xml_response = xml_parser.invoke(response.content)
# print(xml_response)

# CommaSeparatedListOutputParser和 DatetimeOutputParser 输出 了解
ls_parser = CommaSeparatedListOutputParser()
dt_parser = DatetimeOutputParser()
print(ls_parser.get_format_instructions())
print(dt_parser.get_format_instructions())
