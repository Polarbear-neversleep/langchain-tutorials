from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import os
import dotenv

# 复杂检索用retriever

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# 文件加载与切分
text_loader = TextLoader(
    file_path = r"C:\Users\86158\Desktop\科研idea.txt",
    encoding = "utf-8"
)
docs = text_loader.load_and_split()

# 文件切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 0,
    add_start_index = True
)
docs = text_splitter.split_documents(docs)

# 文件嵌入
embed_model = OpenAIEmbeddings(
    model = "text-embedding-3-large"
)

embed_doc = embed_model.embed_documents([doc.page_content for doc in docs])
# 数据存储
db1 = FAISS.from_documents(
    documents= docs,
    embedding= embed_model, 
    # persist_directory="./chroma-1" # 默认存在内存中
)
# 基于向量数据库的检索器
retriever = db1.as_retriever(
    search_type = "mmr", #科研设置多种相似度计算方式
    search_kwargs ={"fetch_k":3}
)
docs = retriever.invoke("边缘计算是什么?")

print(len(docs))
for doc in docs:
    print(doc)
