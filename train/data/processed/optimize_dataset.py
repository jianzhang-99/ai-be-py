"""航运助手训练集约化优化脚本"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

# ============ 1. 读取所有原始数据 ============
data_dir = Path("/Users/liangjiajian/Desktop/AI-dundun/ai-be-py/train/data/processed")
all_samples = []

for fname in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
    fpath = data_dir / fname
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_samples.append(json.loads(line))

print(f"读取样本总数: {len(all_samples)}")

# ============ 2. 槽位修复函数 ============
def fix_slots(query, slots):
    """修复槽位：确保每个 slot 的值都真实出现在 query 中"""
    new_slots = {}
    for key, val in slots.items():
        if val in query:
            new_slots[key] = val
        elif key == "cargo_name":
            cargo_map = {
                "砂石": ["砂石", "砂石料", "碎石"],
                "煤炭": ["煤炭", "煤", "煤泥", "煤矸石", "煤研石"],
                "钢材": ["钢材"],
                "矿石": ["矿石", "锂矿"],
                "水泥": ["水泥"],
                "石子": ["石子"],
            }
            for norm, variants in cargo_map.items():
                if any(v in query for v in variants):
                    new_slots[key] = norm
                    break
        elif key == "cargo_weight":
            wm = re.search(r"(\d+(?:\.\d+)?)\s*(?:吨|万)", query)
            if wm:
                new_slots[key] = wm.group(0)
        elif key == "port_name":
            port_map = {
                "南京港": ["南京港", "南京"],
                "南京龙潭": ["南京龙潭"],
                "重庆果园港": ["重庆果园港", "重庆"],
                "镇江港": ["镇江港", "镇江"],
                "武汉阳逻": ["武汉阳逻"],
                "南通港": ["南通港", "南通"],
            }
            for norm, variants in port_map.items():
                if any(v in query for v in variants):
                    new_slots[key] = norm
                    break
        elif key == "ship_name":
            ship_map = {
                "华航118": ["华航118"],
                "长江之星6号": ["长江之星6号"],
                "江海888": ["江海888"],
                "苏货运01": ["苏货运01"],
                "俞垛79": ["俞垛79"],
            }
            for norm, variants in ship_map.items():
                if any(v in query for v in variants):
                    new_slots[key] = norm
                    break
        elif key in ("route_from", "route_to"):
            if key == "route_from":
                m = re.search(r"([\u4e00-\u9fa5]{1,10}(?:港|口|码头)?)\s*到", query)
            else:
                m = re.search(r"到\s*([\u4e00-\u9fa5]{1,10}(?:港|口|码头)?)", query)
            if m and m.group(1) in query:
                new_slots[key] = m.group(1)
    return new_slots


def fix_sample(sample):
    """修复单个样本"""
    query = sample.get("query", "")
    label = sample.get("label", {})
    fixed_slots = fix_slots(query, label.get("slots", {}))
    sample["label"]["slots"] = fixed_slots
    return sample


# ============ 3. 修复所有样本 ============
fixed_samples = [fix_sample(s) for s in all_samples]
print("槽位修复完成")

# ============ 4. 补充缺失意图样本 ============
new_samples = []

dispatch_samples = [
    {"id": "opt_dispatch_001", "history": [], "query": "华航118现在到哪了，在船上能看到吗", "label": {"intent": "DISPATCH_MONITOR", "slots": {"ship_name": "华航118"}, "need_clarify": False}},
    {"id": "opt_dispatch_002", "history": [], "query": "这条船现在走到哪个航段了", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": True}},
    {"id": "opt_dispatch_003", "history": [], "query": "查看长江之星6号的在途状态", "label": {"intent": "DISPATCH_MONITOR", "slots": {"ship_name": "长江之星6号"}, "need_clarify": False}},
    {"id": "opt_dispatch_004", "history": [], "query": "船舶运输进度帮我盯一下", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": True}},
    {"id": "opt_dispatch_005", "history": [], "query": "江海888现在过安庆了吗", "label": {"intent": "DISPATCH_MONITOR", "slots": {"ship_name": "江海888"}, "need_clarify": False}},
    {"id": "opt_dispatch_006", "history": [], "query": "这票货到哪了，能看实时位置吗", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": True}},
    {"id": "opt_dispatch_007", "history": [], "query": "苏货运01今天能到重庆吗", "label": {"intent": "DISPATCH_MONITOR", "slots": {"ship_name": "苏货运01"}, "need_clarify": False}},
    {"id": "opt_dispatch_008", "history": [], "query": "帮我追踪一下这条船的航行轨迹", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": True}},
    {"id": "opt_dispatch_009", "history": [], "query": "船舶皖航339当前位置在哪", "label": {"intent": "DISPATCH_MONITOR", "slots": {"ship_name": "皖航339"}, "need_clarify": False}},
    {"id": "opt_dispatch_010", "history": [], "query": "货船运输进度怎么查看", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": False}},
]

oil_samples = [
    {"id": "opt_oil_001", "history": [], "query": "南京附近加油站0号柴油价格多少", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "南京", "date_time": "0号"}, "need_clarify": False}},
    {"id": "opt_oil_002", "history": [], "query": "镇江港哪里有加油站", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "镇江港"}, "need_clarify": False}},
    {"id": "opt_oil_003", "history": [], "query": "南通到上海沿线加油站在哪", "label": {"intent": "QUERY_OIL_STATION", "slots": {"route_from": "南通", "route_to": "上海"}, "need_clarify": False}},
    {"id": "opt_oil_004", "history": [], "query": "武汉阳逻港有加油站吗", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "武汉阳逻"}, "need_clarify": False}},
    {"id": "opt_oil_005", "history": [], "query": "问一下最近的加油站位置", "label": {"intent": "QUERY_OIL_STATION", "slots": {}, "need_clarify": True}},
    {"id": "opt_oil_006", "history": [], "query": "太仓港附近0号柴油多少钱一吨", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "太仓港", "date_time": "0号"}, "need_clarify": False}},
    {"id": "opt_oil_007", "history": [], "query": "重庆果园港加油站联系方式有吗", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "重庆果园港"}, "need_clarify": False}},
    {"id": "opt_oil_008", "history": [], "query": "芜湖附近哪里能加油", "label": {"intent": "QUERY_OIL_STATION", "slots": {"port_name": "芜湖"}, "need_clarify": False}},
    {"id": "opt_oil_009", "history": [], "query": "0号柴油今天价格多少", "label": {"intent": "QUERY_OIL_STATION", "slots": {"date_time": "0号"}, "need_clarify": False}},
    {"id": "opt_oil_010", "history": [], "query": "沿江有什么好的加油站推荐吗", "label": {"intent": "QUERY_OIL_STATION", "slots": {}, "need_clarify": False}},
]

ship_info_samples = [
    {"id": "opt_shipinfo_001", "history": [], "query": "查一下华航118的船舶档案", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "华航118"}, "need_clarify": False}},
    {"id": "opt_shipinfo_002", "history": [], "query": "长江之星6号船主是谁", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "长江之星6号"}, "need_clarify": False}},
    {"id": "opt_shipinfo_003", "history": [], "query": "江海888联系方式给我", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "江海888"}, "need_clarify": False}},
    {"id": "opt_shipinfo_004", "history": [], "query": "这条船的证书什么时候到期", "label": {"intent": "QUERY_SHIP_INFO", "slots": {}, "need_clarify": True}},
    {"id": "opt_shipinfo_005", "history": [], "query": "苏货运01船长联系方式有吗", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "苏货运01"}, "need_clarify": False}},
    {"id": "opt_shipinfo_006", "history": [], "query": "船舶吨位和吃水是多少", "label": {"intent": "QUERY_SHIP_INFO", "slots": {}, "need_clarify": True}},
    {"id": "opt_shipinfo_007", "history": [], "query": "俞垛79的船舶证书齐全吗", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "俞垛79"}, "need_clarify": False}},
    {"id": "opt_shipinfo_008", "history": [], "query": "帮我查下这条船的详细信息", "label": {"intent": "QUERY_SHIP_INFO", "slots": {}, "need_clarify": True}},
    {"id": "opt_shipinfo_009", "history": [], "query": "皖航339船主联系电话", "label": {"intent": "QUERY_SHIP_INFO", "slots": {"ship_name": "皖航339"}, "need_clarify": False}},
    {"id": "opt_shipinfo_010", "history": [], "query": "这条船什么时候年检", "label": {"intent": "QUERY_SHIP_INFO", "slots": {}, "need_clarify": True}},
]

clarify_samples = [
    {"id": "opt_clarify_001", "history": [], "query": "帮我查一下船和天气", "label": {"intent": "TALK", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_002", "history": [], "query": "帮我找船", "label": {"intent": "FIND_SHIP", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_003", "history": [], "query": "那条船什么时候到", "label": {"intent": "QUERY_SHIP", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_004", "history": [], "query": "查下我的货到哪了", "label": {"intent": "DISPATCH_MONITOR", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_005", "history": [], "query": "有合适的船吗", "label": {"intent": "FIND_SHIP", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_006", "history": [], "query": "运价多少", "label": {"intent": "QUERY_FREIGHT", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_007", "history": [], "query": "我想查一下这条船", "label": {"intent": "QUERY_SHIP", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_008", "history": [], "query": "帮我查查有没有船", "label": {"intent": "FIND_SHIP", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_009", "history": [], "query": "查下最近的天气", "label": {"intent": "QUERY_WEATHER", "slots": {}, "need_clarify": True}},
    {"id": "opt_clarify_010", "history": [], "query": "船到哪了", "label": {"intent": "QUERY_SHIP", "slots": {}, "need_clarify": True}},
]

talk_samples = [
    {"id": "opt_talk_001", "history": [], "query": "在吗", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_002", "history": [], "query": "帮我查一下", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_003", "history": [], "query": "你们这个怎么用", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_004", "history": [], "query": "早上好", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_005", "history": [], "query": "你好", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_006", "history": [], "query": "安徽在哪", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_007", "history": [], "query": "怎么才能定", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_008", "history": [], "query": "帮我预定这艘船", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_009", "history": [], "query": "浙江有哪些船闸", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
    {"id": "opt_talk_010", "history": [], "query": "航行中要注意哪些安全事项", "label": {"intent": "TALK", "slots": {}, "need_clarify": False}},
]

new_samples.extend(dispatch_samples)
new_samples.extend(oil_samples)
new_samples.extend(ship_info_samples)
new_samples.extend(clarify_samples)
new_samples.extend(talk_samples)

print(f"补充新样本: {len(new_samples)} 条")

# ============ 5. 合并所有样本 ============
all_fixed = fixed_samples + new_samples

# 去重
seen_ids = set()
deduped = []
for s in all_fixed:
    sid = s.get("id", "")
    if sid and sid not in seen_ids:
        seen_ids.add(sid)
        deduped.append(s)

print(f"去重后总样本数: {len(deduped)}")

# ============ 6. 统计意图分布 ============
intent_dist = Counter(s["label"]["intent"] for s in deduped)
print("\n意图分布:")
for intent, cnt in sorted(intent_dist.items(), key=lambda x: -x[1]):
    print(f"  {intent}: {cnt}")

clarify_dist = Counter(s["label"].get("need_clarify", False) for s in deduped)
print(f"\nClarify 分布: {dict(clarify_dist)}")

# ============ 7. 保存合并文件 ============
merged_path = data_dir / "all.jsonl"
with merged_path.open("w", encoding="utf-8") as f:
    for s in deduped:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\n合并文件已保存: {merged_path}")
print(f"总样本数: {len(deduped)}")