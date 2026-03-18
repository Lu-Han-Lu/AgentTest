"""基础计算工具：支持加减乘除、平方、开方等"""
import re

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class CalculateTool(BaseAgentTool):
    def __init__(self):
        self._name = "calculate"
        self._description = """
        用于基础数学计算，支持：
        1. 加减乘除（+、-、*、/）
        2. 平方（如：5的平方）、开方（如：16的开方）
        输入格式示例：100-23.5、5*6+8、9的平方、36的开方
        """

    def _safe_calculate(self, expr: str) -> str:
        """安全计算（仅支持指定运算符，避免安全风险）"""
        # 仅保留允许的字符
        allowed_chars = r'^[0-9\+\-\*\/\.\(\)平方开方 ]+$'
        if not re.match(allowed_chars, expr):
            return "仅支持数字和+、-、*、/、平方、开方运算"

        # 处理平方/开方
        expr = expr.replace("的平方", "**2")
        expr = expr.replace("的开方", "**0.5")

        try:
            # 生产环境建议用 ast.literal_eval 或第三方安全计算库
            result = eval(expr)
            return f"计算结果：{expr} = {result}"
        except ZeroDivisionError:
            return "错误：除数不能为0"
        except Exception as e:
            return f"计算失败：{str(e)[:30]}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        # 清理输入
        clean_query = re.sub(r'计算|等于多少|等于|结果', '', query).strip()
        return self._safe_calculate(clean_query)


# 实例化（可直接注册）
calculate_tool = CalculateTool()