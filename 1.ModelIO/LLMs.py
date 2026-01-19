import os
import dotenv
from langchain_openai import OpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = OpenAI()
str = llm.invoke("写一首诗")
print(str)
print(type(str))
