# demo/langchaintest/myproject0317V1/core/base/llm_base.py
"""统一LLM调用层：供RAG/Agent复用，统一配置/调用/异常处理"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from demo.langchaintest.myproject0317V1.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE, \
    LLM_MAX_TOKENS, LLM_TIMEOUT


class LLMManager:
    """统一LLM管理器：单例模式，避免重复初始化"""
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._llm = cls._init_llm()
        return cls._instance

    @classmethod
    def _init_llm(cls) -> ChatOpenAI:
        """初始化LLM（统一配置）"""
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=LLM_API_KEY,
                openai_api_base=LLM_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS or 4096,
                timeout=LLM_TIMEOUT or 60,
                verbose=False
            )
            return llm
        except Exception as e:
            raise RuntimeError(f"LLM初始化失败：{e}")

    def get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        return self._llm

    def invoke(self, prompt: str, **kwargs) -> str:
        """简化LLM调用接口"""
        llm = self.get_llm()
        chain = RunnablePassthrough() | llm | StrOutputParser()
        return chain.invoke(prompt, **kwargs)

    def batch_invoke(self, prompts: list, **kwargs) -> list:
        """批量调用LLM"""
        llm = self.get_llm()
        chain = RunnablePassthrough() | llm | StrOutputParser()
        return chain.batch(prompts, **kwargs)


# 便捷函数（供原有代码兼容）
def create_llm() -> ChatOpenAI:
    """兼容原有接口，返回LLM实例"""
    return LLMManager().get_llm()
