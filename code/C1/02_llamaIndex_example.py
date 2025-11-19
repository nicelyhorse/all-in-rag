# 练习3：使用 LlamaIndex 复现与 LangChain 类似的 RAG 流程
# ------------------------------------------------------------
# 对应作者在 3.1 中的“初始化设置”
# 这里的 import 与 LangChain 中的初始化作用类似：
# - os / load_dotenv：用来读取环境变量（尤其是 DEEPSEEK_API_KEY）
# - Settings.llm / Settings.embed_model：用来做“全局默认 LLM + 嵌入模型”的配置，
#   相当于在 LangChain 中分别初始化 ChatDeepSeek 和 HuggingFaceEmbeddings，
#   但这里是通过 LlamaIndex 提供的全局 Settings 进行集中管理。

import os
# 如果在国内环境访问 HuggingFace 较慢，可以开启镜像（与 LangChain 部分含义相同）：
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量（对应 3.1 中的“加载环境变量”）
# 主要目的是从 .env 文件中读取 DEEPSEEK_API_KEY 等配置，
# 与 LangChain 示例中的 load_dotenv() 用法完全一致。
load_dotenv()


# ------------------------------------------------------------
# 3.3 索引构建中的“初始化中文嵌入模型”和 LLM 配置（LlamaIndex 版本）
# ------------------------------------------------------------
# 在 LangChain 中，这一步是显式写：
#   embeddings = HuggingFaceEmbeddings(...)
#   llm = ChatDeepSeek(...)
# 在 LlamaIndex 中改为用 Settings.llm 和 Settings.embed_model 做“全局默认”配置：
# - Settings.llm：指定默认使用 DeepSeek 作为大语言模型；
# - Settings.embed_model：指定默认使用 BAAI/bge-small-zh-v1.5 作为文本嵌入模型。
# 后续在 VectorStoreIndex.from_documents 和 query_engine.query 中，
# 都会自动使用这里配置好的 LLM 和 embedding，而不需要再手动传入。

Settings.llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

Settings.embed_model = HuggingFaceEmbedding(
    "BAAI/bge-small-zh-v1.5"
)


# ------------------------------------------------------------
# 3.2 数据准备 (Data Preparation) —— LlamaIndex 版本
# ------------------------------------------------------------
# 对标 LangChain 中的：
#   markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
#   loader = TextLoader(markdown_path)
#   docs = loader.load()
#
# 这里使用 LlamaIndex 提供的 SimpleDirectoryReader：
# - SimpleDirectoryReader 会负责打开文件并封装成 Document 对象；
# - input_files 指定的是单个文件列表，这里直接加载 easy-rl-chapter1.md；
# - 等价于“加载原始文档”这一步的数据准备工作。

docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()


# ------------------------------------------------------------
# 3.3 索引构建 (Index Construction) —— LlamaIndex 版本
# ------------------------------------------------------------
# 对标 LangChain 中的：
#   vectorstore = InMemoryVectorStore(embeddings)
#   vectorstore.add_documents(texts)
#
# 在 LangChain 中，我们显式：
#   1）手动分块（RecursiveCharacterTextSplitter）
#   2）手动调用 embeddings 生成向量
#   3）手动构建 InMemoryVectorStore
#
# 在 LlamaIndex 中，这一行：
#   VectorStoreIndex.from_documents(docs)
# 会自动完成以下工作：
#   - 使用 Settings.embed_model 中配置的 HuggingFaceEmbedding 对文本进行向量化；
#   - 内部执行默认的分块策略（相当于自动完成“文本分块 + 嵌入计算”）；
#   - 构建一个可检索的向量索引对象 index。
# 因此，这句代码整体上对应了 LangChain 示例中“构建向量存储”的那一段逻辑。

index = VectorStoreIndex.from_documents(docs)


# ------------------------------------------------------------
# 3.4 查询与检索 (Query and Retrieval) —— LlamaIndex 版本
# ------------------------------------------------------------
# 对标 LangChain 中的：
#   retrieved_docs = vectorstore.similarity_search(question, k=3)
#   docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
#
# 在 LlamaIndex 中，我们通过 index.as_query_engine() 得到一个 query_engine：
#   - query_engine 内部封装了“相似度检索 + 上下文构造 + 调用 LLM”的完整流程；
#   - 它会自动使用 Settings.llm 和 Settings.embed_model 中的配置；
#   - 使用者只需要调用 query_engine.query("问题") 即可完成“检索 + 生成”一体化操作。
# 因此，下面两句可以视为：
#   - as_query_engine()：对应 LangChain 中“准备检索器 + 准备 prompt”的组合；
#   - query()：对应“similarity_search + prompt.format + llm.invoke(...)”的综合封装。

query_engine = index.as_query_engine()


# 打印当前 query_engine 内部使用的 prompt 模板
# 这一步相当于在 LangChain 中查看 ChatPromptTemplate 的内容，
# 有助于理解 LlamaIndex 默认是如何把“上下文 + 问题”组织成提示词发给 LLM 的。
print(query_engine.get_prompts())


# 实际发起查询：对应 LangChain 中的 question = "文中举了哪些例子？"
# 在这里，query_engine 会：
#   1）先根据问题在向量索引中检索相关的文本块；
#   2）再将检索到的上下文与问题一起打包成 prompt；
#   3）调用 Settings.llm 指定的 DeepSeek 模型生成答案；
#   4）最终返回一个包含 content 的 Response 对象。
print(query_engine.query("文中举了哪些例子?"))
