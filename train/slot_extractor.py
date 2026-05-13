"""
混合槽位抽取模块

结合规则抽取和神经网络抽取：
1. 规则抽取：适用于有强 pattern 的槽位（船名、港口、货物名、重量）
2. 神经网络抽取：BERT BIO 序列标注作为后备

规则抽取的优势：
- 可解释性强，易于调试
- 对数据量要求低
- 对有明确 pattern 的实体效果好

适用槽位类型：
- ship_name: 船舶名（如"华航118"、"长江之星6号"）
- port_name: 港口名（如"南京港"、"重庆果园港"）
- area_name: 区域名（如"长江下游"、"江北"）
- cargo_name: 货物名（如"煤炭"、"砂石"、"钢材"）
- cargo_weight: 重量（如"3000吨"、"1万吨"）
- route_from/route_to: 航线起终点
- date_time: 时间（如"今天"、"月底"、"3月10号"）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 槽位标签定义
SLOT_LABELS = [
    "ship_name",
    "area_name",
    "port_name",
    "route_from",
    "route_to",
    "cargo_name",
    "cargo_weight",
    "date_time",
]

# 船名模式：常见船名后缀和格式
SHIP_PATTERNS = [
    # 船名格式：省/城市 + 货/航/海 + 编号 或 纯编号
    (r"[\u4e00-\u9fa5]{1,4}(?:货|航|海|油|运)[0-9a-zA-Z]{2,6}", "ship_name"),  # 如"苏货运01"、"鲁济宁货6366"
    (r"[\u4e00-\u9fa5]{2,6}(?:号|轮|舶)", "ship_name"),  # 如"华航118号"、"长江之星6号"
    (r"(?:[\u4e00-\u9fa5]{1,4})?[0-9a-zA-Z]{2,6}号", "ship_name"),  # 如"皖航339号"
]

# 港口名模式
PORT_PATTERNS = [
    # 港口名：XX港、XX船闸
    (r"[\u4e00-\u9fa5]+港", "port_name"),
    (r"[\u4e00-\u9fa5]+船闸", "port_name"),
]

# 区域名模式
AREA_PATTERNS = [
    # 区域：XX一带、XX附近、长江XX
    (r"[\u4e00-\u9fa5]+一带", "area_name"),
    (r"[\u4e00-\u9fa5]+附近", "area_name"),
    (r"[\u4e00-\u9fa5]+区域", "area_name"),
    (r"长江[\u4e00-\u9fa5]+", "area_name"),
]

# 货物名模式
CARGO_PATTERNS = [
    # 常见货物
    (r"煤炭|煤|煤泥|原煤|精煤", "cargo_name"),
    (r"砂石|沙子|黄沙|石子|碎石", "cargo_name"),
    (r"钢材|钢筋|钢板|钢管", "cargo_name"),
    (r"水泥|熟料", "cargo_name"),
    (r"矿石|铁矿|铜矿|锰矿", "cargo_name"),
    (r"粮食|大米|小麦|玉米", "cargo_name"),
    (r"木材|原木", "cargo_name"),
    (r"集装箱|货柜", "cargo_name"),
]

# 重量模式（优先级高，因为容易混淆）
WEIGHT_PATTERNS = [
    # 纯数字+单位
    (r"\d{1,3}(?:,?\d{3})*-?\d{0,3}(?:,?\d{3})*吨", "cargo_weight"),
    (r"\d{1,3}(?:,?\d{3})*-?\d{0,3}(?:,?\d{3})*万吨", "cargo_weight"),
    (r"\d{1,3}(?:,?\d{3})*-?\d{0,3}(?:,?\d{3})*公斤", "cargo_weight"),
    # 数字+单位（无逗号）
    (r"\d+吨", "cargo_weight"),
    (r"\d+万吨", "cargo_weight"),
    (r"\d+公斤", "cargo_weight"),
    # 范围值
    (r"\d+-\d+吨", "cargo_weight"),
    (r"\d+-\d+万吨", "cargo_weight"),
    # 约数
    (r"\d+左右吨", "cargo_weight"),
    (r"约\d+吨", "cargo_weight"),
]

# 时间模式
TIME_PATTERNS = [
    # 具体日期
    (r"\d{1,2}月\d{1,2}号", "date_time"),
    (r"\d{1,2}月\d{1,2}日", "date_time"),
    # 月份
    (r"\d{1,2}月", "date_time"),
    (r"本月|上月|下月", "date_time"),
    # 具体时间
    (r"今天|明天|后天", "date_time"),
    (r"上午|下午|早上|晚上|中午", "date_time"),
    (r"\d{1,2}点", "date_time"),
    # 相对时间
    (r"本周|下周|上周", "date_time"),
    (r"本月|下月", "date_time"),
    (r"月底|月中|月初", "date_time"),
    (r"近期|最近|当前", "date_time"),
    (r"\d+天内?", "date_time"),
]

# 航线起终点模式
# 注意：XX和YY应该是地名，不是"有没有"这样的词
ROUTE_PATTERNS = [
    # "从XX到YY" 或 "XX到YY" 或 "XX-YY"
    # 更严格的模式：地名后面是标点、空格、或"有没有"等过渡词
    (r"从([\u4e00-\u9fa5]{1,6})到([\u4e00-\u9fa5]{1,6})", "route_from", "route_to"),
    (r"([\u4e00-\u9fa5]{1,6})到([\u4e00-\u9fa5]{1,6})(?:有没有|有没有|要不要)", "route_from", "route_to"),
    (r"([\u4e00-\u9fa5]{1,6})--([\u4e00-\u9fa5]{1,6})", "route_from", "route_to"),
    (r"([\u4e00-\u9fa5]{1,6})—([\u4e00-\u9fa5]{1,6})", "route_from", "route_to"),
    # 单独的地名识别（当无法确定from/to时）
    (r"[\u4e00-\u9fa5]{1,6}到[\u4e00-\u9fa5]{1,6}", "route_hint", "route_hint"),  # 标记为可能的航线
]

# 柴油类型（加油站相关）
OIL_TYPE_PATTERNS = [
    (r"0号柴油|0#柴油|零号柴油", "oil_type"),
    (r"\d号柴油|\d#柴油", "oil_type"),
]


@dataclass
class ExtractedSlot:
    """抽取到的槽位"""
    slot_type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


class SlotExtractor:
    """
    混合槽位抽取器

    结合规则和神经网络的优势：
    1. 规则抽取：对有明确 pattern 的实体速度快、精度高
    2. 神经网络：处理复杂上下文和未见过的表达
    """

    def __init__(self):
        self.rules = [
            (SHIP_PATTERNS, "ship_name"),
            (PORT_PATTERNS, "port_name"),
            (AREA_PATTERNS, "area_name"),
            (WEIGHT_PATTERNS, "cargo_weight"),
            (TIME_PATTERNS, "date_time"),
            (CARGO_PATTERNS, "cargo_name"),
        ]

    def extract_slots(self, text: str) -> dict[str, list[str]]:
        """
        从文本中抽取槽位

        参数：
            text: 用户输入文本

        返回：
            槽位字典，格式为 {slot_type: [value1, value2, ...]}
        """
        slots: dict[str, list[str]] = {label: [] for label in SLOT_LABELS}
        occupied_ranges: list[tuple[int, int]] = []  # 避免重叠

        def is_overlapping(start: int, end: int) -> bool:
            """检查是否与已抽取范围重叠"""
            for s, e in occupied_ranges:
                if not (end <= s or start >= e):
                    return True
            return False

        def add_slot(slot_type: str, value: str, start: int, end: int):
            """添加槽位（去重且不重叠）"""
            if is_overlapping(start, end):
                return
            # 去重
            if value not in slots[slot_type]:
                slots[slot_type].append(value)
                occupied_ranges.append((start, end))

        # 按优先级处理
        # 1. 航线起终点（需要特殊处理，因为一次匹配两个槽位）
        for pattern, from_slot, to_slot in ROUTE_PATTERNS:
            for match in re.finditer(pattern, text):
                from_val = match.group(1)
                to_val = match.group(2)
                add_slot(from_slot, from_val, match.start(1), match.end(1))
                add_slot(to_slot, to_val, match.start(2), match.end(2))

        # 2. 重量（高优先级，容易与其他实体混淆）
        for pattern, slot_type in WEIGHT_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        # 3. 时间
        for pattern, slot_type in TIME_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        # 4. 船名
        for pattern, slot_type in SHIP_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        # 5. 港口
        for pattern, slot_type in PORT_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        # 6. 区域
        for pattern, slot_type in AREA_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        # 7. 货物名
        for pattern, slot_type in CARGO_PATTERNS:
            for match in re.finditer(pattern, text):
                add_slot(slot_type, match.group(), match.start(), match.end())

        return slots

    def extract_slots_with_confidence(self, text: str) -> list[ExtractedSlot]:
        """返回带置信度的槽位列表"""
        slots = self.extract_slots(text)
        results = []
        for slot_type, values in slots.items():
            for value in values:
                results.append(ExtractedSlot(
                    slot_type=slot_type,
                    text=value,
                    start=0,  # 简化处理
                    end=0,
                    confidence=0.9 if slot_type in ["ship_name", "cargo_weight", "date_time"] else 0.8
                ))
        return results


# 全局实例
slot_extractor = SlotExtractor()


def extract_slots(text: str) -> dict[str, list[str]]:
    """快捷函数：抽取槽位"""
    return slot_extractor.extract_slots(text)


if __name__ == "__main__":
    # 测试
    test_cases = [
        "录入运单，南京龙潭到万州，矿石8000吨，月底装货",
        "帮我查下长江之星6号现在在哪",
        "南京到武汉有没有3000-4000吨的自卸船",
        "重庆到南京有回程船吗，钢材3000吨",
        "苏货运01今天能到重庆吗",
        "查下宜昌到重庆这一段的通航水深",
    ]

    extractor = SlotExtractor()
    for text in test_cases:
        print(f"\n输入: {text}")
        slots = extractor.extract_slots(text)
        for slot_type, values in slots.items():
            if values:
                print(f"  {slot_type}: {values}")
