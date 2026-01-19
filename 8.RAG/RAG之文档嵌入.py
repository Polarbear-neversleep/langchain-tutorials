# 文本加载 -> 文本切分 -> 文本嵌入
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

embedding_model = OpenAIEmbeddings(
    model = "text-embedding-ada-002"
)

# 文本嵌入
# text = "Nice to meet you!"
# embed_query = embedding_model.embed_query(text = text)
# print(len(embed_query))
# print(embed_query[:10])

# 文件嵌入
txt_loader = TextLoader(file_path=r"C:\Users\86158\Desktop\科研idea.txt",encoding = "utf-8")
docs = txt_loader.load_and_split() # 直接载入切分 使用递归切分
embed_docu = embedding_model.embed_documents([doc.page_content for doc in docs])
print(len(embed_docu))
print(len(embed_docu[0]))
print(embed_docu[0][:10])
