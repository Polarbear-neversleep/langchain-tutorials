from langchain_community.document_loaders import TextLoader,PyPDFLoader,CSVLoader,JSONLoader

# 加载txt数据
file_path = "./langchain_introduction.txt"

text_loader = TextLoader(
    file_path=file_path,
    encoding= "utf-8" #字符集格式
)

# docs = text_loader.load()
# print(docs)
# print(docs[0].metadata)
# print(docs[0].page_content)

# 加载pdf数据
pdf_loader = PyPDFLoader(
    file_path=r"D:\xwechat_files\wxid_ygscj0dyqyam22_48df\msg\file\2025-09\2503.21460v1.pdf" #可以是本地或者网上的数据
) 
docs = pdf_loader.load()
# print(len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content)

# 加载csv数据
csv_loader = CSVLoader(
    file_path=r"E:\YOLO11_fine_tuning\runs\detect\train17\results.csv"
)
# docs = csv_loader.load()
# print(len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content)

# 加载json
json_loader = JSONLoader(
    file_path="user.json",
    jq_schema=".users[].name", # .user[]
    text_content=False
)
docs = json_loader.load()
print(docs)
