from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

def create_domain_documents() -> list[Document]:
    """创建包含10个不同领域知识的文档列表"""
    documents = [
        Document(
            page_content="人工智能（Artificial Intelligence）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。其应用领域包括机器学习、计算机视觉、自然语言处理等。",
            metadata={"source": "计算机科学", "type": "技术", "难度": "中等"}
        ),
        Document(
            page_content="量子力学是研究物质世界微观粒子运动规律的物理学分支，主要研究原子、分子、凝聚态物质，以及原子核和基本粒子的结构、性质的基础理论。它与相对论一起构成现代物理学的理论基础。",
            metadata={"source": "物理学", "type": "理论", "难度": "高深"}
        ),
        Document(
            page_content="经济学是研究人类社会在各个发展阶段上的各种经济活动和各种相应的经济关系及其运行、发展的规律的学科。核心思想是物质稀缺性和有效利用资源。",
            metadata={"source": "社会科学", "type": "学科", "难度": "中等"}
        ),
        Document(
            page_content="历史学是人类对自己的历史材料进行筛选和组合的知识形式，是一门研究人类社会过去的学科，通过对历史事件、人物等的研究，总结经验教训，为现实提供借鉴。",
            metadata={"source": "人文科学", "type": "学科", "难度": "基础"}
        ),
        Document(
            page_content="生物学是研究生物（包括植物、动物和微生物）的结构、功能、发生和发展规律的科学，是自然科学的一个门类。其目的在于阐明和控制生命活动，改造自然，为农业、工业和医学等实践服务。",
            metadata={"source": "自然科学", "type": "学科", "难度": "中等"}
        ),
        Document(
            page_content="中医学以阴阳五行作为理论基础，将人体看成是气、形、神的统一体，通过望、闻、问、切四诊合参的方法，探求病因、病性、病位，分析病机及人体内五脏六腑、经络关节、气血津液的变化，判断邪正消长，进而得出病名，归纳出证型，以辨证论治原则，制定治疗方案。",
            metadata={"source": "医学", "type": "传统医学", "难度": "高深"}
        ),
        Document(
            page_content="建筑学是研究建筑及其环境的学科，旨在总结人类建筑活动的经验，以指导建筑设计创作，构造某种体系环境等。它既包含技术科学，也涉及艺术和社会科学。",
            metadata={"source": "工程学", "type": "应用科学", "难度": "中等"}
        ),
        Document(
            page_content="天文学是研究宇宙空间天体、宇宙的结构和发展的学科。内容包括天体的构造、性质和运行规律等。天文学是一门古老的科学，自有人类文明史以来，天文学就有重要的地位。",
            metadata={"source": "自然科学", "type": "基础科学", "难度": "高深"}
        ),
        Document(
            page_content="教育学是研究教育现象和教育问题、揭示教育规律的科学。其研究对象是教育现象和教育问题，任务是揭示教育规律，指导教育实践。",
            metadata={"source": "社会科学", "type": "应用学科", "难度": "基础"}
        ),
        Document(
            page_content="环境科学是研究人类与环境相互关系的科学，研究领域包括自然环境、人工环境和社会环境，目的是揭示环境规律，解决环境问题，协调人与自然的关系。",
            metadata={"source": "交叉学科", "type": "应用科学", "难度": "中等"}
        )
    ]
    return documents

new_docs = create_domain_documents()
# 文件加载与切分
text_loader = TextLoader(
    file_path = r"C:\Users\86158\Desktop\科研idea.txt",
    encoding = "utf-8"
)
docs = text_loader.load_and_split()

# 文件切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 0,
    add_start_index = True
)
new_docs = text_splitter.split_documents(new_docs)

# 文件嵌入
embed_model = OpenAIEmbeddings(
    model = "text-embedding-ada-002"
)

embed_doc = embed_model.embed_documents([doc.page_content for doc in docs])
# 数据存储
db1 = Chroma.from_documents(
    documents= new_docs,
    embedding= embed_model, 
    persist_directory="./chroma-1" # 默认存在内存中
)

# 测试向量数据库
query = "学科难度基础的学科?"
# 使用相似度匹配
docs = db1.similarity_search(query)
# 匹配嵌入向量
embed_query = embed_model.embed_query(query)
docs = db1.similarity_search_by_vector(embed_query)
# 使用L2分数
docs = db1.similarity_search_with_score(query)
# 使用余弦相似度
docs = db1._similarity_search_with_relevance_scores(query,k=3)
# 使用MMR(相似度和多样性的权衡)
for doc,score in docs:
    print(score,doc.page_content)
