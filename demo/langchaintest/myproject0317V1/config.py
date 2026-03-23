# config.py
import os

from dotenv import load_dotenv

# 加载环境变量（如 API 密钥）
load_dotenv()

USER_AGENT = os.environ.get('USER_AGENT',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# 向量库持久化路径
VECTORSTORE_PATH_FAISS = "../faiss_index"
VECTORSTORE_PATH_Chroma = "../chroma_db"
VECTOR_STORE_PATH = "./chroma_db"
FILE_REGISTRY_PATH = "./file_registry.json"
LONG_TERM_MEMORY_PATH = "./long_term_memory"
DOCUMENT_PATH = "./docs/"
SESSION_MEMORY_MAX_ROUNDS = 5

# Embedding 模型配置（可根据需要更换）
EMBEDDING_MODEL_NAME = "D:/Models/BAAI/bge-m3"
# 中文模型可选： "shibing624/text2vec-base-chinese"
EMBEDDING_DEVICE = "cpu"
# 指定本地缓存目录（确保模型已下载，可根据实际路径修改或删除）
CACHE_FOLDER = "C:/Users/h00804919/.cache/huggingface/hub"

# 文本切分配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 检索配置
RETRIEVER_K = 3

# LLM 配置（使用 OpenAI 兼容 API）
LLM_MODEL = os.environ.get("DASHSCOPE_MODEL_ID")
LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 60

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
