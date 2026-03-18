# demo/langchaintest/myproject0317V1/utils/eval_utils.py
"""评估工具：RAGAS/人工评估/效果监控"""
import os
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import pandas as pd
from langchain_core.prompts import PromptTemplate
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_relevancy,
    context_recall, answer_correctness
)
from datasets import Dataset

from demo.langchaintest.myproject0317V1.core.base.llm_base import LLMManager


class RAGEvaluator:
    """RAG评估器：基于RAGAS"""

    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
        self.llm = self.llm_manager.get_llm()

    def evaluate_rag(
            self,
            questions: List[str],
            answers: List[str],
            contexts: List[List[str]],
            ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """评估RAG效果"""
        # 准备数据集
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # 选择评估指标
        metrics = [faithfulness, answer_relevancy, context_relevancy]
        if ground_truths:
            metrics.extend([context_recall, answer_correctness])

        # 执行评估
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                raise_exceptions=False
            )

            # 转换为DataFrame便于分析
            df = result.to_pandas()

            return {
                "metrics": result.scores,  # 整体指标得分
                "detailed": df.to_dict("records"),  # 每条样本的详细得分
                "average": {k: v for k, v in result.scores.items()},  # 平均分
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def save_evaluation_result(self, result: Dict[str, Any], filepath: str):
        """保存评估结果到文件"""
        # 读取已有结果（如果存在）
        existing = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 添加新结果
        existing.append(result)

        # 保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


class AgentEvaluator:
    """Agent评估器：评估工具调用/回答质量"""

    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
        self.eval_prompt = PromptTemplate(
            template="""请从以下维度评估Agent的回答质量：
1. 工具调用必要性：0-5分（0=无需调用，5=必须调用）
2. 工具调用准确性：0-5分（0=调用错误工具，5=调用完全正确）
3. 回答准确性：0-5分（0=完全错误，5=完全正确）
4. 回答完整性：0-5分（0=完全不完整，5=完整覆盖问题）
5. 回答有用性：0-5分（0=完全无用，5=非常有用）

用户问题：{query}
Agent回答：{answer}
工具调用记录：{tool_calls}
标准答案（可选）：{ground_truth}

请按照以下格式输出：
【工具调用必要性】：分数
【工具调用准确性】：分数
【回答准确性】：分数
【回答完整性】：分数
【回答有用性】：分数
【评估说明】：简要说明评分理由

注意：评分必须是0-5的整数，评估说明要具体。""",
            input_variables=["query", "answer", "tool_calls", "ground_truth"]
        )

    def evaluate_agent(
            self,
            query: str,
            answer: str,
            tool_calls: List[Any],
            ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """评估Agent单次回答"""
        # 格式化工具调用记录
        tool_calls_str = "\n".join([
            f"工具：{tc[0].tool}，输入：{tc[0].tool_input}，输出：{tc[1]}"
            for tc in tool_calls or []
        ]) or "无"

        # 构建评估提示
        prompt = self.eval_prompt.format(
            query=query,
            answer=answer,
            tool_calls=tool_calls_str,
            ground_truth=ground_truth or "无"
        )

        # 调用LLM评估
        eval_result = self.llm_manager.invoke(prompt)

        # 解析评分
        scores = self._parse_agent_scores(eval_result)

        return {
            "query": query,
            "answer": answer,
            "tool_calls": tool_calls_str,
            "ground_truth": ground_truth,
            "scores": scores,
            "eval_raw": eval_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _parse_agent_scores(self, eval_result: str) -> Dict[str, float]:
        """解析Agent评分"""
        scores = {
            "tool_call_necessity": 0.0,
            "tool_call_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "answer_completeness": 0.0,
            "answer_usefulness": 0.0,
            "evaluation_note": ""
        }

        lines = eval_result.split("\n")
        for line in lines:
            line = line.strip()
            if "【工具调用必要性】" in line:
                scores["tool_call_necessity"] = self._parse_score(line)
            elif "【工具调用准确性】" in line:
                scores["tool_call_accuracy"] = self._parse_score(line)
            elif "【回答准确性】" in line:
                scores["answer_accuracy"] = self._parse_score(line)
            elif "【回答完整性】" in line:
                scores["answer_completeness"] = self._parse_score(line)
            elif "【回答有用性】" in line:
                scores["answer_usefulness"] = self._parse_score(line)
            elif "【评估说明】" in line:
                scores["evaluation_note"] = line.split("：", 1)[1] if "：" in line else ""

        return scores

    def _parse_score(self, line: str) -> float:
        """解析单个分数"""
        try:
            score_str = line.split("：", 1)[1].strip()
            return float(score_str)
        except (IndexError, ValueError):
            return 0.0

    def batch_evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估Agent"""
        results = []
        total_scores = {
            "tool_call_necessity": 0.0,
            "tool_call_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "answer_completeness": 0.0,
            "answer_usefulness": 0.0
        }

        for data in eval_data:
            result = self.evaluate_agent(
                query=data["query"],
                answer=data["answer"],
                tool_calls=data.get("tool_calls", []),
                ground_truth=data.get("ground_truth")
            )
            results.append(result)

            # 累加总分
            for key in total_scores.keys():
                total_scores[key] += result["scores"][key]

        # 计算平均分
        avg_scores = {k: v / len(results) for k, v in total_scores.items()} if results else total_scores

        return {
            "batch_results": results,
            "average_scores": avg_scores,
            "total_samples": len(results),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }