# 顺序链SimpleSequentialChain 单输入单输出
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain_openai import ChatOpenAI
import os
import dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.llm import LLMChain

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# LLMChain
llm_api = ChatOpenAI(model="gpt-4o-mini")
# llm_loc = ChatOllama(model="deepseek-r1:7b")

prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是生物学专家,精通生物学知识"),
        ("human", "请你详细解释以下现象:{situtation}"),
    ]
)
chain_1 = LLMChain(llm=llm_api, prompt=prompt_1, verbose=True)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "你擅长概括和总结长段内容"),
        ("human", "对以下内容进行精简的概括:{content}"),
    ]
)
chain_2 = LLMChain(llm=llm_api, prompt=prompt_1, verbose=True)

# full_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)
# response = full_chain.invoke(
#     input={"input": "质壁分离的过程"}
# )  # 注意这里只能写input,唯一变量名

# SequentialChain 多输入多输出
# 翻译
prompt_1 = PromptTemplate.from_template(
    "你是一个{field}领域的专家,请你将以下内容翻译成中文:{content}"
)
chain_1 = LLMChain(llm=llm_api, prompt=prompt_1, verbose=True, output_key="translated")
# 总结内容
prompt_2 = PromptTemplate.from_template("对以下内容进行精简的概括:{translated}")
chain_2 = LLMChain(llm=llm_api, prompt=prompt_2, verbose=True, output_key="abstract")
# 提炼关键词
prompt_3 = PromptTemplate.from_template(
    "你是一个{field}领域的专家,请你从以下内容中提取关键词{abstract}"
)
chain_3 = LLMChain(llm=llm_api, prompt=prompt_3, verbose=True, output_key="keywords")

full_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3],
    input_variables=["field", "content"],
    output_variables=["translated", "abstract", "keywords"],
    verbose=True,
)
response = full_chain.invoke(input={"field": "翻译", "content": "Multi-modal AI systems will likely become a ubiquitous presence in our everyday lives. A promising approach to making these systems more interactive is to embody them as agents within physical and virtual environments. At present, systems leverage existing foundation models as the basic building blocks for the creation of embodied agents. Embedding agents within such environments facilitates the ability of models to process and interpret visual and contextual data, which is critical for the creation of more sophisticated and context-aware AI systems. For example, a system that can perceive user actions, human behavior, environmental objects, audio expressions, and the collective sentiment of a scene can be used to inform and direct agent responses within the given environment. To accelerate research on agent-based multimodal intelligence, we define “Agent AI” as a class of interactive systems that can perceive visual stimuli, language inputs, and other environmentallygrounded data, and can produce meaningful embodied actions. In particular, we explore systems that aim to improve agents based on next-embodied action prediction by incorporating external knowledge, multi-sensory inputs, and human feedback. We argue that by developing agentic AI systems in grounded environments, one can also mitigate the hallucinations of large foundation models and their tendency to generate environmentally incorrect outputs. The emerging field of Agent AI subsumes the broader embodied and agentic aspects of multimodal interactions. Beyond agents acting and interacting in the physical world, we envision a future where people can easily create any virtual reality or simulated scene and interact with agents embodied within the virtual environment."})
print(response)
print("翻译后", response["translated"], "\n摘要", response["abstract"], "\n关键词", response["keywords"])