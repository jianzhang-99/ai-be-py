"""
将业务真实话术样本.md 转换为 JSONL 格式

用法：
    python -m train.scripts.import_real_samples
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---- Intent 分类规则 -------------------------------------------------------

def classify_intent(query: str) -> str:
    """根据关键词判断 intent"""
    query = query.strip()

    # 发布运单
    if any(kw in query for kw in ["录一条运单", "帮我建个运单", "录入运单", "我报一条运单", "新建运单",
                                   "这条运单你帮我生成", "发布运单", "航次录入", "上传运单信息", "帮我录一下", "帮你录"]):
        return "SAVE_ORDER"

    # 历史运单
    if any(kw in query for kw in ["历史运单", "运单记录", "之前的运单", "以前的运单", "上次.*运单",
                                   "那票货", "运单查询", "查一下.*运单", "运单.*帮我查"]):
        return "QUERY_ORDER"

    # 查船
    if any(kw in query for kw in ["在哪", "位置", "到哪", "轨迹", "进度", "到港时间", "预计", "现在在",
                                   "帮我查下", "帮我看", "查一下", "看下", "预计几点", "大概还要多久"]):
        return "QUERY_SHIP"

    # 找船
    if any(kw in query for kw in ["找船", "帮我配船", "空船", "驳船", "货船", "船舶分布", "附近的船",
                                   "周围.*船", "有没有.*船", "有没有5000", "帮我找.*船"]):
        return "FIND_SHIP"

    # 运价
    if any(kw in query for kw in ["运价", "运费", "多少钱", "价格", "大概是多少", "优惠"]):
        return "QUERY_FREIGHT"

    # 天气
    if any(kw in query for kw in ["天气", "雨", "风", "台风", "能见度", "暴雨", "降温", "温度"]):
        return "QUERY_WEATHER"

    # 水位/水深
    if any(kw in query for kw in ["水位", "水深", "吃水", "通航", "航道"]):
        return "QUERY_WATER_LEVEL"

    # 反馈
    if any(kw in query for kw in ["建议", "投诉", "闪退", "不好用", "打不通", "体验", "更新后"]):
        return "FEEDBACK"

    # 图片识别
    if any(kw in query for kw in ["照片", "帮我识别", "提取.*文字", "图片"]):
        return "IMAGE_OCR"

    # 文档问答
    if any(kw in query for kw in ["详细介绍", "你能做什么", "怎么注册", "注册需要", "介绍一下"]):
        return "DOC_QA"

    # 闲聊
    if any(kw in query for kw in ["你好", "在吗", "是什么模型", "航行.*注意", "安全法规", "船员",
                                   "船闸", "过闸", "加油", "柴油", "安徽", "浙江", "帮我预定"]):
        return "TALK"

    return "TALK"


# ---- Slot 抽取规则 -------------------------------------------------------

SHIP_NAMES = {
    "长江之星6号", "华航118", "华航018", "江海888", "苏货运01",
    "皖航339", "浙海运66", "金顺168", "兴隆09", "俞垛79",
    "鲁济宁货6366", "苏盐城货209388", "济宁", "兴隆09",
}

PORT_NAMES = {
    "南京龙潭", "南京港", "重庆果园港", "镇江港", "武汉阳逻",
    "南通港", "如皋港", "江阴港", "张家港", "宜昌港",
    "万州港", "九江港", "扬州港", "太仓港", "上海外高桥",
    "武穴码头", "如皋港务", "湖口", "彭泽", "淮安",
    "盱眙", "蒙城", "铜陵", "颍上", "涡阳",
    "黄冈", "黄石", "芜湖", "淮南", "嘉定", "泰州",
}

CARGO_NAMES = {
    "砂石": "砂石", "砂石料": "砂石料", "煤炭": "煤炭", "钢材": "钢材",
    "矿石": "矿石", "水泥": "水泥", "玉米": "玉米", "石子": "石子",
    "煤矸石": "煤矸石", "煤矸": "煤矸", "煤研石": "煤研石",
    "锂矿": "锂矿", "大豆": "大豆", "碎石": "碎石",
    "大理石": "大理石", "煤泥": "煤泥", "抛江石": "抛江石",
    "沙子": "沙子", "煤": "煤",
}

def extract_slots(query: str) -> dict:
    """从 query 中提取槽位"""
    slots = {}

    # 精确匹配船名
    for ship in SHIP_NAMES:
        if ship in query:
            slots["ship_name"] = ship
            break

    # 精确匹配港口
    for port in PORT_NAMES:
        if port in query:
            slots["port_name"] = port
            break

    # 精确匹配货名
    for cargo_key, cargo_val in CARGO_NAMES.items():
        if cargo_key in query:
            # 槽位标注要保留原句中的表层词，否则 BIO 对齐找不到文本位置。
            slots["cargo_name"] = cargo_val
            break

    # 匹配吨位
    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:吨|万)", query)
    if weight_match:
        slots["cargo_weight"] = weight_match.group(0)

    # 匹配航线 from
    route_from_match = re.search(r"([\u4e00-\u9fa5]{1,6}(?:港|口|码头)?)\s*到", query)
    if route_from_match:
        slots["route_from"] = route_from_match.group(1).strip()

    # 匹配航线 to
    route_to_match = re.search(r"到\s*([\u4e00-\u9fa5]{1,6}(?:港|口|码头|果园)?)", query)
    if route_to_match:
        slots["route_to"] = route_to_match.group(1).strip()

    # 匹配时间
    date_patterns = [
        r"(\d+[月日号]\d*[日号]?)",
        r"(\d+日|\d+号|月底|月初|明天|后天|今天|下周|本月|本周|这周|那周|装货|装期)",
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, query)
        if date_match:
            slots["date_time"] = date_match.group(1).strip()
            break

    return slots


def parse_sample_file(filepath: str) -> list[dict]:
    """解析样本文件，返回样本列表"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    samples = []
    sample_id = 0

    # 按 ### 分割
    parts = re.split(r"\n###\s+", content)

    for part in parts[1:]:  # 跳过第一个空部分
        lines = part.strip().split("\n")
        section_title = lines[0].strip()

        # 解析样本行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # 跳过标题行和空行
            if line.startswith("#") or line.startswith("---") or line.startswith("|") or line.startswith("**"):
                continue

            # 跳过列表标记和序号
            line_clean = re.sub(r"^[\-\d\.\s\*]+", "", line).strip()
            if not line_clean or len(line_clean) < 2:
                continue

            # 跳过明显的标题行
            if len(line_clean) > 100:
                continue

            sample_id += 1
            query = line_clean
            intent = classify_intent(query)
            slots = extract_slots(query)

            samples.append({
                "id": f"real_{sample_id:05d}",
                "history": [],
                "query": query,
                "label": {
                    "intent": intent,
                    "slots": slots,
                    "need_clarify": False,
                },
                "metadata": {
                    "section": section_title,
                }
            })

    return samples


def main():
    source_file = "/Users/liangjiajian/Desktop/AI-dundun/ai-be-py/doc/样本集合/业务真实话术样本.md"
    output_dir = "/train/data/realCase"

    samples = parse_sample_file(source_file)

    # 按 intent 分组
    intent_groups: dict[str, list[dict]] = {}
    for sample in samples:
        intent = sample["label"]["intent"]
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(sample)

    print(f"\n解析完成，共 {len(samples)} 条样本")
    print("意图分布：")
    for intent, sms in sorted(intent_groups.items(), key=lambda x: -len(x[1])):
        with_slots = sum(1 for s in sms if s["label"]["slots"])
        print(f"  {intent}: {len(sms)} 条 (有槽位: {with_slots})")

    # 保存所有样本到一个文件
    all_path = f"{output_dir}/全部样本.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\n✓ 保存全部样本到 {all_path}")

    # 按 intent 分组保存
    for intent, sms in intent_groups.items():
        intent_file = f"{output_dir}/{intent}.jsonl"
        with open(intent_file, "w", encoding="utf-8") as f:
            for sample in sms:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✓ 保存 {intent} 样本到 {intent_file}")

    # 合并到训练集
    merged_path = f"{output_dir}/训练集_真实样本.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✓ 保存合并训练集到 {merged_path}")

    # 显示示例
    print("\n示例样本:")
    for intent in ["QUERY_SHIP", "FIND_SHIP", "SAVE_ORDER"]:
        if intent in intent_groups:
            s = intent_groups[intent][0]
            print(f"  [{intent}] {s['query']} -> slots: {s['label']['slots']}")


if __name__ == "__main__":
    main()
