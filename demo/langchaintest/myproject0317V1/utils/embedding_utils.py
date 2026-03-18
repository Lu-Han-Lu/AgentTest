# demo/langchaintest/myproject0317V1/utils/embedding_utils.py
from langchain_huggingface import HuggingFaceEmbeddings

from demo.langchaintest.myproject0317V1.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, CACHE_FOLDER


def create_embeddings(model_name: str = None, device: str = None) -> HuggingFaceEmbeddings:
    """创建Embedding模型实例"""
    model_name = model_name or EMBEDDING_MODEL_NAME
    device = device or EMBEDDING_DEVICE

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device, 'local_files_only': True},
        cache_folder=CACHE_FOLDER,
    )
