# demo/langchaintest/myproject0317V1/utils/llm_utils.py
"""LLM相关工具函数：输入准备、答案清理、结果展示等"""
import re
from typing import Dict, Any, List


def prepare_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备输入给 Prompt 的数据。
    输入: {"question": 用户问题, "docs": 检索到的文档列表}
    输出: 包含格式化上下文、来源列表、原始文档的字典
    """
    question = inputs["question"]
    docs = inputs["docs"]  # List[Document]

    # 格式化上下文（带来源标注）
    formatted_context = "\n\n---\n\n".join([
        f"[来源: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    ])

    # 提取去重后的来源文件名
    sources = list(set([doc.metadata['source'] for doc in docs]))

    return {
        "question": question,
        "context": formatted_context,
        "sources": sources,
        "retrieved_docs": docs
    }


def clean_answer(answer: str) -> str:
    """清理答案中的 think 标签等无关内容"""
    # 移除 <think> ... </think> 及其前后空白
    return re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL).strip()


def display_result(q: str, result: Dict[str, Any]):
    """打印查询结果，包含答案、来源和参考内容片段"""
    answer = clean_answer(result["answer"])
    sources = result["sources"]
    retrieved_docs = result["retrieved_docs"]

    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"来源文件: {', '.join(sources) if sources else '无'}")

    print("\n参考内容片段:")
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            preview = doc.page_content.strip().replace('\n', ' ')[:150]
            if len(doc.page_content) > 150:
                preview += "..."
            print(f"  [{i + 1}] 来源: {doc.metadata['source']}")
            print(f"      内容: {preview}")
    else:
        print("  (未检索到相关文档)")
    print("=" * 60)