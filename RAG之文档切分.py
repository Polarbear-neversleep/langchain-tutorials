# 文档切分
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_community.document_loaders import TextLoader

text = "这是第一段内容。这是第二段文本\n这是最后一段\n\n"

# 基类TextSplitter
# 字符文本分词器：依赖固定字符，允许重叠，分隔符优先
text_splitter = CharacterTextSplitter(
    separator="。", # 分隔符优先
    chunk_size=20, #最大字节数 真实场景chunk_size 4000左右，chunk_overlap在10%到20%
    chunk_overlap=6, # 重叠大小
    keep_separator=True # 保留分隔符,默认False
)

chunks = text_splitter.split_text(text) # 单段字符串分词用split_text

# for i,chunk in enumerate(chunks):
#     print(f"块 {i+1} :长度:{len(chunk)}")
#     print(chunk)

# 递归文本切分器(最常用)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 8,
    add_start_index = True # 加索引，只用docunment才能用str不行
) # 循环分隔符 ["\n\n", "\n", " ", ""]

txt_loader = TextLoader(file_path=r"C:\Users\86158\Desktop\科研idea.txt",encoding = "utf-8")
docs = txt_loader.load()

chunks = text_splitter.split_text(text)
# 使用create_document (要求字符串列表，生成document列表)
chunks = text_splitter.create_documents([text])
# 使用split_document (要求输入document列表，使用文件加载器)
chunks = text_splitter.split_documents(docs)

# print(type(chunks[0]))
# print(len(chunks))
# for chunk in chunks:
#     print(f"🔥:",chunk.page_content)

# 使用token拆分,会优先兼顾到自然边界，与LLM token计数一致，尽量保证语义完整性
text_splitter = TokenTextSplitter(
    chunk_size = 40, #限制token数目
    chunk_overlap = 0,
    encoding_name ="cl100k_base" #将文本转为token序列(OpenAI编码器)
)
chunks = text_splitter.split_documents(docs)
print(len(chunks))
for chunk in chunks:
    print(f"🔥:",chunk.page_content)

# 语义分块方法 SemanticChunker，同样需要嵌入模型判断前后文语义
# breakpoint_threshold_type 判断断点阈值类型
# breakpoint_threshold_amount 判断断点阈值大小