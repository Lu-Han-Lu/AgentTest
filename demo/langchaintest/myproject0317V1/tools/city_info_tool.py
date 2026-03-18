"""城市基础信息工具：查询人口、面积、所属省份、特产等"""
import re

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class CityInfoTool(BaseAgentTool):
    def __init__(self):
        self._name = "city_info"
        self._description = """
        用于查询国内城市基础信息，支持：
        1. 人口、面积、所属省份
        2. 特产、著名景点
        3. 简称、电话区号
        输入格式示例：北京人口、上海的特产、广州的著名景点
        """
        # Mock 城市数据
        self.city_data = {
            "北京": {
                "人口": "2184.3万人（2023年）",
                "面积": "16410.54平方公里",
                "省份": "直辖市",
                "特产": "北京烤鸭、景泰蓝、茯苓夹饼",
                "景点": "故宫、天安门、长城、颐和园",
                "简称": "京",
                "区号": "010"
            },
            "上海": {
                "人口": "2487.09万人（2023年）",
                "面积": "6340.5平方公里",
                "省份": "直辖市",
                "特产": "小笼包、生煎包、大白兔奶糖",
                "景点": "外滩、东方明珠、迪士尼、豫园",
                "简称": "沪",
                "区号": "021"
            },
            "广州": {
                "人口": "1873.41万人（2023年）",
                "面积": "7434.4平方公里",
                "省份": "广东省",
                "特产": "肠粉、叉烧包、广式月饼",
                "景点": "广州塔、长隆、陈家祠、沙面",
                "简称": "穗",
                "区号": "020"
            },
            "深圳": {
                "人口": "1766.18万人（2023年）",
                "面积": "1997.47平方公里",
                "省份": "广东省",
                "特产": "沙井生蚝、公明烧鹅、南山荔枝",
                "景点": "世界之窗、欢乐谷、莲花山、大梅沙",
                "简称": "深",
                "区号": "0755"
            }
        }

    def _get_city_info(self, query: str) -> str:
        """解析并返回城市信息"""
        # 提取城市和查询维度
        city_pattern = r"(北京|上海|广州|深圳)"
        city_match = re.search(city_pattern, query)
        if not city_match:
            return "暂支持查询：北京、上海、广州、深圳的信息"

        city = city_match.group()
        info_type = ""
        if "人口" in query:
            info_type = "人口"
        elif "面积" in query:
            info_type = "面积"
        elif "省份" in query or "所属" in query:
            info_type = "省份"
        elif "特产" in query:
            info_type = "特产"
        elif "景点" in query or "旅游" in query:
            info_type = "景点"
        elif "简称" in query:
            info_type = "简称"
        elif "区号" in query or "电话" in query:
            info_type = "区号"
        else:
            # 返回全部信息
            info = self.city_data[city]
            return f"""【{city}基础信息】
人口：{info['人口']}
面积：{info['面积']}
所属：{info['省份']}
特产：{info['特产']}
景点：{info['景点']}
简称：{info['简称']}
区号：{info['区号']}"""

        # 返回指定维度信息
        return f"{city}的{info_type}：{self.city_data[city][info_type]}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        return self._get_city_info(query)


# 实例化（可直接注册）
city_info_tool = CityInfoTool()