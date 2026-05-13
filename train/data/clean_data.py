"""
数据清洗脚本 - 阶段一：数据边界重建
功能：读取原始case数据，清洗脏样本，按query去重后输出train/valid/test
"""
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# 意图标签定义
INTENT_LABELS = [
    "DOC_QA", "FIND_SHIP", "SAVE_ORDER", "QUERY_ORDER", "QUERY_SHIP",
    "QUERY_FREIGHT", "QUERY_WEATHER", "QUERY_WATER_LEVEL", "DISPATCH_MONITOR",
    "IMAGE_OCR", "FEEDBACK", "QUERY_OIL_STATION", "QUERY_SHIP_INFO", "TALK"
]

# 真正的闲聊query（只保留这些作为TALK）
TRUE_TALK_QUERIES = {
    "你好", "早上好", "在吗", "你们这个怎么用", "帮我查一下", "你是谁",
    "你是什么模型", "你是用的什么ai模型", "太仓武港到南钢需要开船开多久",
    "过淮安船闸排队要多久", "航行中要注意哪些安全事项", "安全法规有哪些",
    "我想考船员适任证书在哪里考", "安徽在哪", "帮我看一下，从芜湖到淮南，我需要过几道闸"
}

# 需要修正的TALK样本（这些实际上是业务查询）
TALK_CORRECTIONS = {
    "real_00016": "QUERY_SHIP",  # 华航118什么时候到港
    "real_00038": "QUERY_SHIP",  # 苏盐城货209388什么时候到装货港
    "real_00040": "QUERY_SHIP",  # 现在淮安船闸上行有多少条船在等待
    "real_00150": "QUERY_ORDER",  # 麻烦查下南京装的砂石运单历史
    "real_00152": "QUERY_ORDER",  # 帮我查去年10月南京到武汉的运单
    "real_00154": "QUERY_ORDER",  # 以前从南京港走矿石到宜昌的运单有吗
    "real_00149": "QUERY_ORDER",  # 查一下南京到重庆去年下半年的运单
    "real_00214": "QUERY_SHIP",  # 过淮安船闸排队要多久
    "real_00200": "IMAGE_OCR",  # 识别图中文字
    "real_00201": "IMAGE_OCR",  # 提取图中文字
    "real_00212": "QUERY_SHIP",  # 我想考船员适任证书在哪里考
    "real_00219": "QUERY_SHIP",  # 安徽在哪
    "real_00224": "QUERY_SHIP",  # 帮我看一下，从芜湖到淮南，我需要过几道闸
    "real_00225": "QUERY_SHIP",  # 帮我查一下闸口信息
    # 找船相关的都是FIND_SHIP
    "real_00041": "FIND_SHIP",
    "real_00042": "FIND_SHIP",
    "real_00043": "FIND_SHIP",
    "real_00044": "FIND_SHIP",
    "real_00045": "FIND_SHIP",
    "real_00046": "FIND_SHIP",
    "real_00047": "FIND_SHIP",
    "real_00048": "FIND_SHIP",
    "real_00049": "FIND_SHIP",
    "real_00050": "FIND_SHIP",
    "real_00051": "FIND_SHIP",
    "real_00052": "FIND_SHIP",
    "real_00053": "FIND_SHIP",
    "real_00054": "FIND_SHIP",
    "real_00055": "FIND_SHIP",
    "real_00056": "FIND_SHIP",
    "real_00057": "FIND_SHIP",
    "real_00058": "FIND_SHIP",
    "real_00059": "FIND_SHIP",
    "real_00060": "FIND_SHIP",
    "real_00061": "FIND_SHIP",
    "real_00062": "FIND_SHIP",
    "real_00063": "FIND_SHIP",
    "real_00064": "FIND_SHIP",
    "real_00065": "FIND_SHIP",
    "real_00066": "FIND_SHIP",
    "real_00067": "FIND_SHIP",
    "real_00068": "FIND_SHIP",
    "real_00069": "FIND_SHIP",
    "real_00070": "FIND_SHIP",
    "real_00071": "FIND_SHIP",
    "real_00074": "FIND_SHIP",
    "real_00077": "FIND_SHIP",
    "real_00082": "FIND_SHIP",
    "real_00083": "FIND_SHIP",
    "real_00087": "FIND_SHIP",
    "real_00088": "FIND_SHIP",
    "real_00090": "FIND_SHIP",
    "real_00091": "FIND_SHIP",
    "real_00093": "FIND_SHIP",
    "real_00094": "FIND_SHIP",
    "real_00095": "FIND_SHIP",
    "real_00097": "FIND_SHIP",
    "real_00098": "FIND_SHIP",
    "real_00099": "FIND_SHIP",
    "real_00100": "FIND_SHIP",
    "real_00101": "FIND_SHIP",
    "real_00102": "FIND_SHIP",
    "real_00103": "FIND_SHIP",
    "real_00105": "FIND_SHIP",
    "real_00108": "FIND_SHIP",
    "real_00112": "FIND_SHIP",
    "real_00113": "FIND_SHIP",
    "real_00114": "FIND_SHIP",
    "real_00116": "FIND_SHIP",
    "real_00118": "FIND_SHIP",
    "real_00121": "FIND_SHIP",
    "real_00123": "FIND_SHIP",
    "real_00125": "FIND_SHIP",
    "real_00129": "FIND_SHIP",
    "real_00132": "FIND_SHIP",
    # 天气相关
    "real_00165": "QUERY_WEATHER",  # 帮我看下南京到重庆这条线这两天天气怎么样
    "real_00169": "QUERY_WEATHER",  # 查一下安庆附近水域今晚天气
    "real_00171": "QUERY_WEATHER",  # 帮我看下南京港本周天气趋势
    # 水位相关
    "real_00183": "QUERY_WATER_LEVEL",  # 查一下南京附近航道水位
    "real_00191": "QUERY_WATER_LEVEL",  # 帮我看下镇江到南京这段通航水深是否正常
    # 运价相关
    "real_00213": "QUERY_FREIGHT",  # 运价能优惠吗？
    "real_00215": "QUERY_FREIGHT",  # 召集加油站0号柴油多少钱一吨
}

# 垃圾槽位值检测
GARBAGE_SLOT_PATTERNS = [
    r"^什么时候$",
    r"^港$",
    r"^哪了$",
    r"^哪段了$",
    r"^船$",
    r"^就装$",
    r"^随$",
    r"^随装$",
    r"^哪里了$",
    r"^港时间帮我看$",
    r"^重庆果园港大$",
    r"^号走$",
    r"^6号$",
    r"^装货$",
    r"^本月$",
    r"^这段航线明$",
    r"^这条线这$",
    r"^武汉运力$",
    r"^重庆煤炭$",
    r"^芜湖有没有$",
    r"^武汉的船$",
    r"^南通一德码头明天装$",
    r"^南通",
    r"^京博通科技园",
    r"^上海宝山区三",
    r"^宜昌的驳船",
    r"^武汉运力",
    r"^重庆下半",
    r"^武汉这条线最",
    r"^重庆主航道水",
    r"^九江这条线水",
    r"^芜湖这一段现",
    r"^镇江这段航道",
    r"^镇江通航水深",
    r"^查下宜昌",
    r"^月南京",
    # 修复route_from/route_to过宽问题：排除功能词/上下文词
    r"^帮我",
    r"^预计",
    r"^查询",
    r"^我想查",
    # 过短或无意义的词
    r"^现在$",
    r"^这里$",
    r"^那边$",
    r"^这条$",
    r"^那条$",
    # 被截断的片段（包含查询词或介词残骸）
    r"^位置和",
    r"^号什么时候",
    r"^查一下",
    r"^查下",
]


def is_garbage_slot(value: str) -> bool:
    """检测是否是垃圾槽位值"""
    if not value:
        return True
    if len(value) <= 1:
        return True
    for pattern in GARBAGE_SLOT_PATTERNS:
        if re.match(pattern, value):
            return True
    return False


def clean_slots(slots: Dict) -> Dict:
    """清洗槽位，移除垃圾值"""
    cleaned = {}
    valid_keys = {"ship_name", "area_name", "port_name", "route_from", "route_to",
                  "cargo_name", "cargo_weight", "date_time"}
    for k, v in slots.items():
        if k in valid_keys and not is_garbage_slot(v):
            cleaned[k] = v
    return cleaned


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(path: str, samples: List[Dict]) -> None:
    """保存JSONL文件"""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def stats_summary(samples: List[Dict], title: str = "") -> Dict:
    """生成统计信息"""
    intent_count = defaultdict(int)
    clarify_pos = 0
    clarify_neg = 0

    for s in samples:
        intent = s.get("label", {}).get("intent", "UNKNOWN")
        intent_count[intent] += 1
        if s.get("label", {}).get("need_clarify", False):
            clarify_pos += 1
        else:
            clarify_neg += 1

    return {
        "title": title,
        "total": len(samples),
        "intent_dist": dict(intent_count),
        "clarify_pos": clarify_pos,
        "clarify_neg": clarify_neg,
        "clarify_ratio": f"{clarify_pos}/{clarify_pos + clarify_neg}"
    }


def main():
    # ============================================================
    # 第一步：读取case目录下的原始数据
    # ============================================================
    case_dir = "E:/ai-be-py/train/data/case"
    processed_dir = "E:/ai-be-py/train/data/processed"

    # 各case文件的意图分布统计
    case_stats = {}

    # 读取发布运单（检查重复）
    save_order_raw = load_jsonl(f"{case_dir}/发布运单_cleaned.jsonl")
    print(f"发布运单原始: {len(save_order_raw)}条")

    # ============================================================
    # 第二步：合并所有case数据并清洗
    # ============================================================
    all_case_samples = []
    seen_queries = set()

    # 定义case文件到意图的映射
    case_intent_map = {
        "查船_cleaned.jsonl": "QUERY_SHIP",
        "闲聊_cleaned.jsonl": "TALK",
        "运价查询_cleaned.jsonl": "QUERY_FREIGHT",
        "反馈投诉_cleaned.jsonl": "FEEDBACK",
        "发布运单_cleaned.jsonl": "SAVE_ORDER",
        "水位查询_cleaned.jsonl": "QUERY_WATER_LEVEL",
        "天气查询_cleaned.jsonl": "QUERY_WEATHER",
        "文档问答_cleaned.jsonl": "DOC_QA",
        "图片识别_cleaned.jsonl": "IMAGE_OCR",
        "找船_cleaned.jsonl": "FIND_SHIP",
        "订单查询_cleaned.jsonl": "QUERY_ORDER",
    }

    dirty_samples = []  # 记录被清洗掉的脏样本
    corrected_samples = []  # 记录被修正的样本

    for filename, default_intent in case_intent_map.items():
        filepath = f"{case_dir}/{filename}"
        samples = load_jsonl(filepath)

        for s in samples:
            query = s.get("query", "")
            label = s.get("label", {})
            intent = label.get("intent", default_intent)
            slots = label.get("slots", {})

            # 修正被错误标注的TALK样本
            sample_id = s.get("id", "")
            if sample_id in TALK_CORRECTIONS:
                corrected_intent = TALK_CORRECTIONS[sample_id]
                dirty_samples.append({
                    "id": sample_id,
                    "query": query,
                    "original_intent": intent,
                    "corrected_intent": corrected_intent,
                    "reason": "TALK误标为业务意图"
                })
                intent = corrected_intent

            # TALK类只保留真正的闲聊
            if intent == "TALK":
                is_true_talk = False
                for true_talk in TRUE_TALK_QUERIES:
                    if true_talk in query:
                        is_true_talk = True
                        break
                if not is_true_talk and query not in TRUE_TALK_QUERIES:
                    # 检查是否是业务查询
                    if any(kw in query for kw in ["查", "找", "船", "货", "运", "订单", "历史"]):
                        dirty_samples.append({
                            "id": sample_id,
                            "query": query,
                            "original_intent": "TALK",
                            "corrected_intent": "FILTERED",
                            "reason": "TALK包含业务内容但非真正闲聊"
                        })
                        continue  # 跳过这类样本

            # 清洗槽位
            cleaned_slots = clean_slots(slots)

            # 清理后的样本
            cleaned_sample = {
                "id": s.get("id", ""),
                "history": s.get("history", []),
                "query": query,
                "label": {
                    "intent": intent,
                    "slots": cleaned_slots,
                    "need_clarify": label.get("need_clarify", False)
                }
            }

            # 按query去重（仅对无history的单轮样本）
            if not s.get("history") and query:
                if query in seen_queries:
                    continue
                seen_queries.add(query)

            all_case_samples.append(cleaned_sample)

    print(f"\ncase数据合并后: {len(all_case_samples)}条")
    print(f"清洗掉的脏样本: {len(dirty_samples)}条")
    print(f"被修正的样本: {len(corrected_samples)}条")

    # ============================================================
    # 第三步：读取processed数据并合并
    # ============================================================

    # 读取all.jsonl - 这是主要的合并数据源
    all_processed = load_jsonl(f"{processed_dir}/all.jsonl")
    print(f"\nall.jsonl: {len(all_processed)}条")

    # 读取clarify_samples - 需要clarify的样本
    clarify_samples = load_jsonl(f"{processed_dir}/clarify_samples.jsonl")
    print(f"clarify_samples: {len(clarify_samples)}条")

    # 读取longtail_samples - 长尾样本
    longtail_samples = load_jsonl(f"{processed_dir}/longtail_samples.jsonl")
    print(f"longtail_samples: {len(longtail_samples)}条")

    # 多轮样本只能从all.jsonl的multiturn子集中拆分，且必须保证与单轮test/valid互斥
    # all.jsonl没有history字段，所以多轮数据实际全部来自multiturn_test
    # 正确做法：multiturn_train和multiturn_test从同一个多轮池子按8:2拆分，id不重叠

    # 读取多轮原始数据（从原始备份读取，不依赖运行后状态）
    # 如果multiturn_test.jsonl已经被之前的运行切小（只剩10条），说明是后续运行，需要跳过本步骤
    multiturn_raw_path = f"{processed_dir}/multiturn_test.jsonl"
    import os
    if os.path.getsize(multiturn_raw_path) < 5000:  # 原始50条约10KB，切小后只有2KB左右
        print(f"multiturn_test已被切小，跳过多轮数据处理（这是后续运行）")
        mt_train_samples = []
        mt_test_samples = []
    else:
        with open(multiturn_raw_path, "r", encoding="utf-8") as f:
            import json
            multiturn_raw = [json.loads(line) for line in f if line.strip()]
        print(f"multiturn原始: {len(multiturn_raw)}条")

        # 拆分：前40条给train，后10条给test（id不重叠）
        mt_train_samples = multiturn_raw[:40]
        mt_test_samples = multiturn_raw[40:]

    # 确保train和test的id不重叠
    train_ids = set(s['id'] for s in mt_train_samples)
    test_ids = set(s['id'] for s in mt_test_samples)
    assert len(train_ids & test_ids) == 0, "多轮train/test ID重叠！"

    # 从已有的valid/test文件读取（这些是历史积累的单轮评估集）
    # 注意：多轮样本的valid/test是独立的，来自mt_test_samples的拆分
    existing_valid = load_jsonl(f"{processed_dir}/valid_cleaned.jsonl")
    existing_test = load_jsonl(f"{processed_dir}/test_cleaned.jsonl")
    print(f"existing_valid: {len(existing_valid)}条")
    print(f"existing_test: {len(existing_test)}条")

    # ============================================================
    # 第四步：构建最终的train/valid/test数据集
    # ============================================================

    # 先对valid/test去重（保证评估集之间没有重叠）
    all_eval_queries = {}  # query -> source (valid/test/both)
    for s in existing_valid:
        q = s.get("query", "")
        if q not in all_eval_queries:
            all_eval_queries[q] = {"valid": s}
        else:
            all_eval_queries[q]["valid"] = s  # 保留最后一个
    for s in existing_test:
        q = s.get("query", "")
        if q not in all_eval_queries:
            all_eval_queries[q] = {"test": s}
        else:
            all_eval_queries[q]["test"] = s  # 保留最后一个

    # 按优先級分配：valid保留，test去重
    final_valid = []
    final_test = []
    assigned_queries = set()

    for q, sources in all_eval_queries.items():
        if "valid" in sources and "test" in sources:
            # 两者重叠：查哪个后出现的就保留到哪个集
            # 这里优先保留valid，把test的去重掉
            pass  # 两个都在，走下面分配逻辑
        if q not in assigned_queries:
            if "valid" in sources:
                final_valid.append(sources["valid"])
                assigned_queries.add(q)
            elif "test" in sources:
                final_test.append(sources["test"])
                assigned_queries.add(q)

    valid_samples = final_valid
    test_samples = final_test

    train_samples = []
    train_queries = set()

    # 先添加case样本
    for s in all_case_samples:
        query = s.get("query", "")
        if query and query not in train_queries:
            train_samples.append(s)
            train_queries.add(query)

    # 添加clarify_samples（这些需要clarify的作为训练数据）
    for s in clarify_samples:
        query = s.get("query", "")
        if query and query not in train_queries:
            train_samples.append(s)
            train_queries.add(query)

    # 添加longtail_samples
    for s in longtail_samples:
        query = s.get("query", "")
        if query and query not in train_queries:
            train_samples.append(s)
            train_queries.add(query)

    # 添加mt_train_samples到训练集（多轮训练）
    for s in mt_train_samples:
        # 多轮样本用history+query组合来去重
        key = f"{json.dumps(s.get('history', []), ensure_ascii=True)}|{s.get('query', '')}"
        if key not in train_queries:
            train_samples.append(s)
            train_queries.add(key)

    # ============================================================
    # 第五步：保存清洗后的数据
    # ============================================================
    output_dir = "E:/ai-be-py/train/data/processed"

    save_jsonl(f"{output_dir}/train_cleaned.jsonl", train_samples)
    save_jsonl(f"{output_dir}/valid_cleaned.jsonl", valid_samples)
    save_jsonl(f"{output_dir}/test_cleaned.jsonl", test_samples)
    save_jsonl(f"{output_dir}/multiturn_train.jsonl", mt_train_samples)
    save_jsonl(f"{output_dir}/multiturn_test.jsonl", mt_test_samples)

    # ============================================================
    # 第六步：输出详细统计信息
    # ============================================================

    train_stat = stats_summary(train_samples, "训练集")
    valid_stat = stats_summary(valid_samples, "验证集")
    test_stat = stats_summary(test_samples, "测试集")
    mt_train_stat = stats_summary(mt_train_samples, "多轮训练集")
    mt_test_stat = stats_summary(mt_test_samples, "多轮测试集")

    print("\n========== 各文件统计 ==========")

    for stat in [train_stat, valid_stat, test_stat, mt_train_stat, mt_test_stat]:
        print(f"\n【{stat['title']}】共 {stat['total']} 条")
        print(f"  Intent分布: {stat['intent_dist']}")
        print(f"  Clarify正负: {stat['clarify_ratio']}")

    print("\n========== 清洗掉的脏样本 ==========")
    print(f"共 {len(dirty_samples)} 条脏样本:")
    for d in dirty_samples[:20]:
        print(f"  [{d['id']}] {d['query'][:30]}... | 原:{d['original_intent']} -> 修:{d['corrected_intent']} | {d['reason']}")
    if len(dirty_samples) > 20:
        print(f"  ... 还有 {len(dirty_samples) - 20} 条")

    # 统计脏样本原因
    reason_count = defaultdict(int)
    for d in dirty_samples:
        reason_count[d["reason"]] += 1
    print(f"\n脏样本原因分析:")
    for reason, count in reason_count.items():
        print(f"  {reason}: {count}条")


if __name__ == "__main__":
    main()