"""待办事项管理工具：添加/查询/删除/清空待办"""
import json
import os
import re

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class TodoTool(BaseAgentTool):
    def __init__(self, save_path: str = "./todo_list.json"):
        self._name = "todo_manage"
        self._description = """
        用于管理待办事项，支持：
        1. 添加待办：添加待办 开会
        2. 查询待办：查询所有待办、查询待办数量
        3. 删除待办：删除待办 1（删除第1条）
        4. 清空待办：清空所有待办
        数据本地存储，重启后不丢失
        """
        self.save_path = save_path
        # 初始化待办文件
        self._init_todo_file()

    def _init_todo_file(self):
        """初始化待办文件（不存在则创建）"""
        if not os.path.exists(self.save_path):
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_todos(self) -> list:
        """加载待办列表"""
        with open(self.save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_todos(self, todos: list):
        """保存待办列表"""
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)

    def _manage_todo(self, query: str) -> str:
        """处理待办管理请求"""
        todos = self._load_todos()

        # 添加待办
        if "添加待办" in query:
            todo_content = re.sub(r'添加待办|添加|待办', '', query).strip()
            if not todo_content:
                return "请输入待办内容，示例：添加待办 下午3点开会"
            todos.append(todo_content)
            self._save_todos(todos)
            return f"已添加待办（第{len(todos)}条）：{todo_content}"

        # 查询待办
        elif "查询待办" in query or "查看待办" in query:
            if not todos:
                return "暂无待办事项"
            if "数量" in query:
                return f"当前待办数量：{len(todos)}条"
            # 格式化输出
            todo_str = "\n".join([f"{i + 1}. {todo}" for i, todo in enumerate(todos)])
            return f"【当前待办列表】\n{todo_str}"

        # 删除待办
        elif "删除待办" in query:
            if not todos:
                return "暂无待办事项可删除"
            # 提取待办序号
            num_pattern = r'删除待办 (\d+)'
            num_match = re.search(num_pattern, query)
            if not num_match:
                return "请指定要删除的待办序号，示例：删除待办 1"
            idx = int(num_match.group(1)) - 1  # 转0索引
            if idx < 0 or idx >= len(todos):
                return f"待办序号错误，当前只有{len(todos)}条待办"
            deleted = todos.pop(idx)
            self._save_todos(todos)
            return f"已删除待办：{deleted}"

        # 清空待办
        elif "清空待办" in query:
            self._save_todos([])
            return "已清空所有待办事项"

        else:
            return "支持操作：添加待办、查询待办、删除待办、清空待办"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        return self._manage_todo(query)


# 实例化（可直接注册）
todo_tool = TodoTool()