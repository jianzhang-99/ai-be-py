"""
数据诊断脚本

每次训练前自动输出数据集统计信息：
- intent 分布
- clarify 正负比例
- 每个 slot 出现次数
- 每个 slot 的平均 span 长度
- 标注值未命中 query 的样本

用法：
    python -m train.scripts.data_diagnostics --data_dir ./train/data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from train.common import (
    BIO_LABELS,
    INTENT_LABELS,
    SLOT_LABELS,
    build_model_text,
    find_slot_spans,
    normalize_slot_values,
    read_jsonl,
    validate_samples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_intent_distribution(samples: list[dict]) -> dict:
    """分析 intent 分布"""
    intent_counts = Counter()
    for sample in samples:
        intent = sample.get("label", {}).get("intent", "UNKNOWN")
        intent_counts[intent] += 1

    total = len(samples)
    result = {
        "total": total,
        "distribution": {
            intent: {
                "count": count,
                "ratio": round(count / total * 100, 2) if total > 0 else 0,
            }
            for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])
        },
        "missing_intents": [i for i in INTENT_LABELS if i not in intent_counts],
    }
    return result


def analyze_clarify_distribution(samples: list[dict]) -> dict:
    """分析 clarify 正负比例"""
    positive = 0
    negative = 0
    no_clarify_question = 0  # need_clarify=true 但没有 clarify_question

    for sample in samples:
        label = sample.get("label", {})
        need_clarify = label.get("need_clarify", False)
        clarify_question = label.get("clarify_question", "").strip()

        if need_clarify:
            positive += 1
            if not clarify_question:
                no_clarify_question += 1
        else:
            negative += 1

    total = len(samples)
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_ratio": round(positive / total * 100, 2) if total > 0 else 0,
        "negative_ratio": round(negative / total * 100, 2) if total > 0 else 0,
        "missing_clarify_question_count": no_clarify_question,
    }


def analyze_slot_distribution(samples: list[dict]) -> dict:
    """分析每个 slot 出现次数"""
    slot_counts = Counter()
    slot_values_counter = defaultdict(Counter)  # slot -> value -> count
    slot_span_lengths = defaultdict(list)  # slot -> [span_length, ...]
    empty_slot_count = 0

    for sample in samples:
        label = sample.get("label", {})
        slots = label.get("slots", {})
        text = build_model_text(sample)

        for slot_name in SLOT_LABELS:
            values = normalize_slot_values(slots.get(slot_name))
            if not values:
                continue

            slot_counts[slot_name] += len(values)

            for value in values:
                slot_values_counter[slot_name][value] += 1

                # 计算 span 长度
                span_len = len(value)
                slot_span_lengths[slot_name].append(span_len)

                # 检查是否在 text 中
                if value not in text:
                    logger.warning(f"Slot {slot_name}={value} 未出现在 text 中: {text[:50]}...")

        # 检查空槽位
        for slot_name, values in slots.items():
            if slot_name in SLOT_LABELS and not normalize_slot_values(values):
                empty_slot_count += 1

    result = {
        "total_slots": sum(slot_counts.values()),
        "empty_slot_count": empty_slot_count,
        "slot_counts": dict(sorted(slot_counts.items(), key=lambda x: -x[1])),
        "slot_span_avg_length": {
            slot: round(sum(lengths) / len(lengths), 2) if lengths else 0
            for slot, lengths in slot_span_lengths.items()
        },
        "top_values": {
            slot: dict(counter.most_common(5))
            for slot, counter in slot_values_counter.items()
        },
    }
    return result


def analyze_bio_distribution(samples: list[dict]) -> dict:
    """分析 BIO 标签分布"""
    bio_counts = Counter()
    total_bio_tags = 0

    for sample in samples:
        label = sample.get("label", {})
        slots = label.get("slots", {})
        text = build_model_text(sample)

        # 构建 BIO 标签
        spans = find_slot_spans(text, slots)
        bio_tags = ["O"] * len(text)
        for start, end, slot_name, value in spans:
            bio_tags[start] = f"B-{slot_name}"
            for i in range(start + 1, end):
                bio_tags[i] = f"I-{slot_name}"

        for tag in bio_tags:
            if tag != "O":
                bio_counts[tag] += 1
                total_bio_tags += 1

    # 计算 O 占比
    total_chars = sum(len(build_model_text(s)) for s in samples)
    o_count = total_chars - total_bio_tags
    o_ratio = round(o_count / total_chars * 100, 2) if total_chars > 0 else 0

    return {
        "total_chars": total_chars,
        "o_count": o_count,
        "o_ratio": o_ratio,
        "entity_count": total_bio_tags,
        "entity_ratio": round(100 - o_ratio, 2),
        "bio_distribution": dict(bio_counts),
    }


def analyze_missed_slots(samples: list[dict]) -> list[dict]:
    """找出标注值未命中 query 的样本"""
    missed = []
    for sample in samples:
        label = sample.get("label", {})
        slots = label.get("slots", {})
        text = build_model_text(sample)

        missed_slots = []
        for slot_name, raw_value in slots.items():
            if slot_name not in SLOT_LABELS:
                continue
            values = normalize_slot_values(raw_value)
            for value in values:
                if value and value not in text:
                    missed_slots.append({
                        "slot": slot_name,
                        "value": value,
                    })

        if missed_slots:
            missed.append({
                "id": sample.get("id", ""),
                "query": sample.get("query", ""),
                "missed_slots": missed_slots,
            })

    return missed


def analyze_context_inheritance(samples: list[dict]) -> dict:
    """分析多轮对话上下文继承情况"""
    has_history = 0
    has_context_inherited = 0

    for sample in samples:
        history = sample.get("history", [])
        if history:
            has_history += 1

        if sample.get("label", {}).get("context_inherited", False):
            has_context_inherited += 1

    return {
        "total": len(samples),
        "has_history": has_history,
        "has_history_ratio": round(has_history / len(samples) * 100, 2) if samples else 0,
        "has_context_inherited": has_context_inherited,
        "context_inherited_ratio": round(has_context_inherited / has_history * 100, 2) if has_history else 0,
    }


def print_diagnostics_report(
    intent_stats: dict,
    clarify_stats: dict,
    slot_stats: dict,
    bio_stats: dict,
    missed_samples: list[dict],
    context_stats: dict,
) -> None:
    """打印诊断报告"""
    print("\n" + "=" * 70)
    print("数据集诊断报告")
    print("=" * 70)

    # Intent 分布
    print("\n【Intent 分布】")
    print(f"  总样本数: {intent_stats['total']}")
    print(f"  缺失 Intent: {', '.join(intent_stats['missing_intents']) if intent_stats['missing_intents'] else '无'}")
    print("  分布:")
    for intent, info in intent_stats['distribution'].items():
        bar = "█" * int(info['ratio'] / 2) + "░" * (50 - int(info['ratio'] / 2))
        print(f"    {intent:20s} [{bar}] {info['count']:4d} ({info['ratio']:5.2f}%)")

    # Clarify 分布
    print("\n【Clarify 分布】")
    print(f"  总样本数: {clarify_stats['total']}")
    print(f"  正样本: {clarify_stats['positive']} ({clarify_stats['positive_ratio']:.2f}%)")
    print(f"  负样本: {clarify_stats['negative']} ({clarify_stats['negative_ratio']:.2f}%)")
    if clarify_stats['missing_clarify_question_count'] > 0:
        print(f"  ⚠️ 警告: {clarify_stats['missing_clarify_question_count']} 条 need_clarify=true 但无 clarify_question")

    # Slot 分布
    print("\n【Slot 分布】")
    print(f"  总槽位数: {slot_stats['total_slots']}")
    print(f"  空槽位标注: {slot_stats['empty_slot_count']}")
    print("  各 Slot 出现次数:")
    for slot, count in slot_stats['slot_counts'].items():
        bar = "█" * int(count / 10) + "░" * max(0, 20 - int(count / 10))
        print(f"    {slot:15s} [{bar}] {count}")

    print("  各 Slot 平均 Span 长度:")
    for slot, avg_len in slot_stats['slot_span_avg_length'].items():
        print(f"    {slot:15s}  {avg_len:.2f}")

    print("  各 Slot Top5 值:")
    for slot, values in slot_stats['top_values'].items():
        print(f"    {slot:15s}  {values}")

    # BIO 分布
    print("\n【BIO 标签分布】")
    print(f"  总字符数: {bio_stats['total_chars']}")
    print(f"  O 标签: {bio_stats['o_count']} ({bio_stats['o_ratio']:.2f}%)")
    print(f"  实体标签: {bio_stats['entity_count']} ({bio_stats['entity_ratio']:.2f}%)")
    print("  实体标签分布:")
    for bio_tag, count in sorted(bio_stats['bio_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {bio_tag:15s}  {count}")

    # 未命中样本
    if missed_samples:
        print(f"\n【⚠️ 未命中样本】({len(missed_samples)} 条)")
        for sample in missed_samples[:10]:
            print(f"  ID: {sample['id']}")
            print(f"    Query: {sample['query'][:50]}...")
            print(f"    Missed: {sample['missed_slots']}")
    else:
        print("\n【未命中样本】无")

    # 上下文继承
    print("\n【多轮对话上下文】")
    print(f"  有历史轮次: {context_stats['has_history']} ({context_stats['has_history_ratio']:.2f}%)")
    print(f"  上下文继承: {context_stats['has_context_inherited']} ({context_stats['context_inherited_ratio']:.2f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="数据诊断脚本")
    parser.add_argument("--data_dir", type=str, default="./train/data/processed", help="数据目录")
    parser.add_argument("--file", type=str, default="all.jsonl", help="数据文件名")
    parser.add_argument("--output", type=str, default="", help="诊断结果输出路径")
    args = parser.parse_args()

    data_path = Path(args.data_dir) / args.file
    if not data_path.exists():
        logger.error("数据文件不存在: %s", data_path)
        sys.exit(1)

    logger.info("加载数据: %s", data_path)
    samples = read_jsonl(data_path)
    logger.info("样本数: %s", len(samples))

    # 执行各项分析
    logger.info("分析 Intent 分布...")
    intent_stats = analyze_intent_distribution(samples)

    logger.info("分析 Clarify 分布...")
    clarify_stats = analyze_clarify_distribution(samples)

    logger.info("分析 Slot 分布...")
    slot_stats = analyze_slot_distribution(samples)

    logger.info("分析 BIO 分布...")
    bio_stats = analyze_bio_distribution(samples)

    logger.info("检查未命中样本...")
    missed_samples = analyze_missed_slots(samples)

    logger.info("分析上下文继承...")
    context_stats = analyze_context_inheritance(samples)

    # 打印报告
    print_diagnostics_report(
        intent_stats,
        clarify_stats,
        slot_stats,
        bio_stats,
        missed_samples,
        context_stats,
    )

    # 保存结果
    results = {
        "intent_stats": intent_stats,
        "clarify_stats": clarify_stats,
        "slot_stats": slot_stats,
        "bio_stats": bio_stats,
        "missed_samples": missed_samples,
        "context_stats": context_stats,
    }

    output_path = Path(args.output) if args.output else data_path.parent / "diagnostics_report.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("诊断报告已保存到: %s", output_path)


if __name__ == "__main__":
    main()