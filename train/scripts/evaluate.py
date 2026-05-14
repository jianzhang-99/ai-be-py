"""
意图识别模型评测脚本

覆盖核心离线指标：
- Intent Accuracy / Macro F1 / Weighted F1
- Clarify Precision / Recall / F1
- Slot Entity Precision / Recall / F1
- Exact Match
- Context Inheritance Rate

用法：
    python -m train.scripts.evaluate \
        --model_path ./train/outputs/checkpoints/best_model.bin \
        --test_file ./train/data/processed/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train.common import (
    BIO_LABELS,
    ID_TO_INTENT,
    INTENT_LABELS,
    MODEL_NAME,
    IntentDataset,
    build_model_text,
    confusion_matrix_counts,
    decode_bio_entities,
    entity_key,
    gold_slot_entities,
    read_jsonl,
    safe_div,
)
from train.scripts.train import BertIntentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 中文意图标签说明
INTENT_LABELS_CN = {
    "DOC_QA": "文档问答",
    "FIND_SHIP": "找船",
    "SAVE_ORDER": "发布运单",
    "QUERY_ORDER": "查询订单",
    "QUERY_SHIP": "查船位置",
    "QUERY_FREIGHT": "运价查询",
    "QUERY_WEATHER": "天气查询",
    "QUERY_WATER_LEVEL": "水位查询",
    "DISPATCH_MONITOR": "在途监控",
    "IMAGE_OCR": "图片识别",
    "FEEDBACK": "反馈投诉",
    "QUERY_OIL_STATION": "加油站查询",
    "QUERY_SHIP_INFO": "船舶档案",
    "TALK": "闲聊",
}


def classification_metrics(preds: list[int], labels: list[int], label_names: list[str]) -> dict[str, Any]:
    """不依赖 sklearn 的多分类 Accuracy / Macro F1 / Weighted F1。"""
    total = len(labels)
    correct = sum(1 for pred, label in zip(preds, labels) if pred == label)
    support = Counter(labels)
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    weighted_f1_sum = 0.0

    for idx, name in enumerate(label_names):
        tp = sum(1 for pred, label in zip(preds, labels) if pred == idx and label == idx)
        fp = sum(1 for pred, label in zip(preds, labels) if pred == idx and label != idx)
        fn = sum(1 for pred, label in zip(preds, labels) if pred != idx and label == idx)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        label_support = support.get(idx, 0)

        per_label[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": label_support,
            "name_cn": INTENT_LABELS_CN.get(name, name),
        }
        f1_values.append(f1)
        weighted_f1_sum += f1 * label_support

    return {
        "accuracy": safe_div(correct, total),
        "f1_macro": safe_div(sum(f1_values), len(f1_values)),
        "f1_weighted": safe_div(weighted_f1_sum, total),
        "per_label": per_label,
    }


def slot_entity_metrics(pred_entities: list[list[dict]], gold_entities: list[list[dict]]) -> dict[str, float]:
    """按完整实体计算槽位 Precision / Recall / F1。"""
    tp = fp = fn = 0
    for pred_list, gold_list in zip(pred_entities, gold_entities):
        pred_set = {entity_key(entity) for entity in pred_list}
        gold_set = {entity_key(entity) for entity in gold_list}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_model(model, dataloader, samples: list[dict[str, Any]], tokenizer, device: torch.device) -> dict[str, Any]:
    """在测试集上评测模型，并保留少量错误样例方便人工复盘。"""
    model.eval()

    intent_preds: list[int] = []
    intent_labels: list[int] = []
    clarify_preds: list[int] = []
    clarify_labels: list[int] = []
    slot_preds: list[list[dict]] = []
    slot_golds: list[list[dict]] = []
    bad_cases: list[dict[str, Any]] = []

    sample_cursor = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_ids = batch["intent_id"].to(device)
            clarify_ids = batch["clarify_id"].to(device)

            intent_logits, slot_logits, clarify_logits = model(input_ids, attention_mask)
            batch_intent_preds = intent_logits.argmax(dim=-1).cpu().tolist()
            batch_clarify_preds = clarify_logits.argmax(dim=-1).cpu().tolist()
            batch_slot_preds = slot_logits.argmax(dim=-1).cpu().tolist()

            intent_preds.extend(batch_intent_preds)
            intent_labels.extend(intent_ids.cpu().tolist())
            clarify_preds.extend(batch_clarify_preds)
            clarify_labels.extend(clarify_ids.cpu().tolist())

            for row_idx, bio_ids in enumerate(batch_slot_preds):
                sample = samples[sample_cursor]
                text = build_model_text(sample)
                encoding = tokenizer(
                    text,
                    max_length=input_ids.size(1),
                    padding="max_length",
                    truncation=True,
                    return_offsets_mapping=True,
                )
                pred_entities = decode_bio_entities(bio_ids, encoding["offset_mapping"], text)
                gold_entities = gold_slot_entities(sample)
                slot_preds.append(pred_entities)
                slot_golds.append(gold_entities)

                pred_intent = ID_TO_INTENT.get(batch_intent_preds[row_idx], "UNKNOWN")
                gold_intent = sample.get("label", {}).get("intent", "UNKNOWN")
                pred_clarify = bool(batch_clarify_preds[row_idx])
                gold_clarify = bool(sample.get("label", {}).get("need_clarify", False))
                slot_ok = {entity_key(e) for e in pred_entities} == {entity_key(e) for e in gold_entities}

                if (pred_intent != gold_intent or pred_clarify != gold_clarify or not slot_ok) and len(bad_cases) < 50:
                    bad_cases.append(
                        {
                            "id": sample.get("id"),
                            "query": sample.get("query"),
                            "gold": {
                                "intent": gold_intent,
                                "need_clarify": gold_clarify,
                                "slots": [entity_key(e) for e in gold_entities],
                            },
                            "pred": {
                                "intent": pred_intent,
                                "need_clarify": pred_clarify,
                                "slots": [entity_key(e) for e in pred_entities],
                            },
                        }
                    )
                sample_cursor += 1

    intent_metrics = classification_metrics(intent_preds, intent_labels, INTENT_LABELS)

    clarify_counts = confusion_matrix_counts(clarify_labels, clarify_preds)
    clarify_precision = safe_div(clarify_counts.get("tp", 0), clarify_counts.get("tp", 0) + clarify_counts.get("fp", 0))
    clarify_recall = safe_div(clarify_counts.get("tp", 0), clarify_counts.get("tp", 0) + clarify_counts.get("fn", 0))
    clarify_f1 = safe_div(2 * clarify_precision * clarify_recall, clarify_precision + clarify_recall)

    slot_metrics = slot_entity_metrics(slot_preds, slot_golds)

    exact_match_count = 0
    intent_slots_match_count = 0
    context_total = 0
    context_correct = 0
    per_intent = defaultdict(lambda: {"correct": 0, "total": 0, "exact_match": 0})
    # 按 intent 统计 slot entity f1
    per_intent_slot_f1: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    # 按 intent 统计 clarify recall（只统计 need_clarify=True 的）
    per_intent_clarify_recall: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, sample in enumerate(samples):
        gold_intent = sample.get("label", {}).get("intent", "UNKNOWN")
        pred_intent = ID_TO_INTENT.get(intent_preds[idx], "UNKNOWN")
        pred_slot_set = {entity_key(e) for e in slot_preds[idx]}
        gold_slot_set = {entity_key(e) for e in slot_golds[idx]}
        intent_ok = pred_intent == gold_intent
        slots_ok = pred_slot_set == gold_slot_set
        gold_clarify = sample.get("label", {}).get("need_clarify", False)
        pred_clarify = bool(clarify_preds[idx])
        clarify_ok = pred_clarify == gold_clarify

        per_intent[gold_intent]["total"] += 1
        if intent_ok:
            per_intent[gold_intent]["correct"] += 1
        if intent_ok and slots_ok:
            intent_slots_match_count += 1
        if intent_ok and slots_ok and clarify_ok:
            exact_match_count += 1
            per_intent[gold_intent]["exact_match"] += 1

        # 统计 clarify recall（只看 need_clarify=True 的正样本）
        if gold_clarify:
            per_intent_clarify_recall[gold_intent]["total"] += 1
            if clarify_ok:
                per_intent_clarify_recall[gold_intent]["correct"] += 1

        # 统计 per-intent slot entity f1
        for entity in slot_preds[idx]:
            if entity in gold_slot_set:
                per_intent_slot_f1[gold_intent]["tp"] += 1
            else:
                per_intent_slot_f1[gold_intent]["fp"] += 1
        for entity in gold_slot_set:
            if entity not in pred_slot_set:
                per_intent_slot_f1[gold_intent]["fn"] += 1

        if sample.get("label", {}).get("context_inherited"):
            context_total += 1
            if intent_ok and slots_ok:
                context_correct += 1

    total = len(samples)

    # 计算 business-oriented score
    # score = 0.35 * intent_f1_macro + 0.25 * clarify_f1 + 0.25 * slot_f1 + 0.15 * exact_match
    business_score = (
        0.35 * intent_metrics["f1_macro"]
        + 0.25 * clarify_f1
        + 0.25 * slot_metrics["f1"]
        + 0.15 * safe_div(exact_match_count, total)
    )

    return {
        "intent": intent_metrics,
        "clarify": {
            "precision": clarify_precision,
            "recall": clarify_recall,
            "f1": clarify_f1,
            "counts": clarify_counts,
        },
        "slot_entity": slot_metrics,
        "end_to_end": {
            "exact_match": safe_div(exact_match_count, total),
            "intent_slots_match": safe_div(intent_slots_match_count, total),
            "context_inheritance_rate": safe_div(context_correct, context_total),
            "context_total": context_total,
        },
        "business_score": business_score,
        "per_intent_accuracy": {
            intent: {
                "accuracy": safe_div(stats["correct"], stats["total"]),
                "correct": stats["correct"],
                "total": stats["total"],
                "exact_match": stats["exact_match"],
                "exact_match_ratio": safe_div(stats["exact_match"], stats["total"]),
            }
            for intent, stats in sorted(per_intent.items())
        },
        "per_intent_slot_f1": {
            intent: {
                "f1": safe_div(2 * d["tp"], 2 * d["tp"] + d["fp"] + d["fn"]) if (2 * d["tp"] + d["fp"] + d["fn"]) > 0 else 0,
                "precision": safe_div(d["tp"], d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0,
                "recall": safe_div(d["tp"], d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0,
                "tp": d["tp"],
                "fp": d["fp"],
                "fn": d["fn"],
            }
            for intent, d in sorted(per_intent_slot_f1.items())
        },
        "per_intent_clarify_recall": {
            intent: {
                "recall": safe_div(stats["correct"], stats["total"]) if stats["total"] > 0 else 0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for intent, stats in sorted(per_intent_clarify_recall.items())
        },
        "bad_cases": bad_cases,
    }


def load_metadata_model_name(model_path: Path) -> str:
    """优先从 checkpoint 目录 metadata.json 读取预训练模型名。"""
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.exists():
        return MODEL_NAME
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata.get("model_name", MODEL_NAME)


def save_badcase_to_dir(bad_cases: list[dict[str, Any]], badcase_dir: Path, intent_labels: list[str]) -> None:
    """
    将 bad cases 按意图分类保存到 JSONL 文件。

    每个意图一行，文件名如 FIND_SHIP.jsonl，共用同一目录。
    如果某个意图没有 bad cases，则不生成对应文件。
    """
    badcase_dir.mkdir(parents=True, exist_ok=True)

    # 按意图分组
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in bad_cases:
        gold_intent = case.get("gold", {}).get("intent", "UNKNOWN")
        by_intent[gold_intent].append(case)

    # 追加写入每个意图的 JSONL 文件
    for intent, cases in by_intent.items():
        intent_file = badcase_dir / f"{intent}.jsonl"
        # 追加模式，防止覆盖历史版本积累的数据
        mode = "a" if intent_file.exists() else "w"
        with intent_file.open(mode, encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    logger.info("Bad cases 已保存到 %s (%s 个意图)", badcase_dir, len(by_intent))


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 意图识别评测脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="", help="评估结果输出路径，默认写到测试集同目录")
    parser.add_argument("--badcase_dir", type=str, default="", help="Bad cases 输出目录，默认保存到 train/data/badcase")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_path = Path(args.test_file)
    output_path = Path(args.output_file) if args.output_file else test_path.parent / "evaluation_results.json"
    model_name = load_metadata_model_name(model_path)

    logger.info("评测模型: %s", model_path)
    logger.info("测试数据: %s", test_path)
    logger.info("预训练模型: %s", model_name)

    device = torch.device(args.device)
    tokenizer_dir = model_path.parent if (model_path.parent / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

    model = BertIntentClassifier(
        model_name=model_name,
        num_intents=len(INTENT_LABELS),
        num_bio_labels=len(BIO_LABELS),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    samples = read_jsonl(test_path)
    test_dataset = IntentDataset(test_path, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    results = evaluate_model(model, test_loader, samples, tokenizer, device)

    # 中文输出结果
    intent_m = results["intent"]
    clarify_m = results["clarify"]
    slot_m = results["slot_entity"]
    end_to_end = results["end_to_end"]

    print("\n" + "=" * 70)
    print("意图识别模型评测报告")
    print("=" * 70)

    # 业务综合评分
    business_score = results.get("business_score", 0)
    print(f"\n【业务综合评分】 {business_score:.4f}")
    print("  (score = 0.35*intent_f1_macro + 0.25*clarify_f1 + 0.25*slot_f1 + 0.15*exact_match)")

    print("\n【意图识别】")
    print(f"  准确率 (Accuracy): {intent_m['accuracy']:.4f}")
    print(f"  Macro F1:          {intent_m['f1_macro']:.4f}")
    print(f"  Weighted F1:       {intent_m['f1_weighted']:.4f}")

    print("\n【澄清判断】")
    print(f"  Precision: {clarify_m['precision']:.4f}")
    print(f"  Recall:    {clarify_m['recall']:.4f}")
    print(f"  F1:        {clarify_m['f1']:.4f}")

    print("\n【槽位抽取】")
    print(f"  Entity F1:       {slot_m['f1']:.4f}")
    print(f"  Entity Recall:   {slot_m['recall']:.4f}")
    print(f"  Entity Prec:     {slot_m['precision']:.4f}")

    print("\n【端到端匹配】")
    print(f"  Exact Match:            {end_to_end['exact_match']:.4f}")
    print(f"  Intent+Slots Match:     {end_to_end['intent_slots_match']:.4f}")
    print(f"  Context Inheritance:    {end_to_end['context_inheritance_rate']:.4f}")

    # 各意图详细指标
    print("\n【各意图详细指标】")
    per_intent_acc = results.get("per_intent_accuracy", {})
    per_intent_slot_f1 = results.get("per_intent_slot_f1", {})
    per_intent_clarify_recall = results.get("per_intent_clarify_recall", {})

    print(f"  {'意图':<12} {'准确率':>8} {'ExactMatch':>10} {'Slot F1':>8} {'Clarify Recall':>12} {'样本数':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")

    for intent in INTENT_LABELS:
        stats = per_intent_acc.get(intent, {"accuracy": 0, "correct": 0, "total": 0, "exact_match": 0})
        slot_f1_info = per_intent_slot_f1.get(intent, {"f1": 0})
        clarify_info = per_intent_clarify_recall.get(intent, {"recall": 0, "total": 0})

        cn_name = INTENT_LABELS_CN.get(intent, intent)
        acc = stats["accuracy"]
        em = stats.get("exact_match_ratio", safe_div(stats.get("exact_match", 0), stats["total"]))
        sf1 = slot_f1_info.get("f1", 0)
        cr = clarify_info.get("recall", 0)
        total = stats["total"]

        bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
        print(f"  {cn_name:<12} [{bar}] {acc:.2%}   {em:.2%}       {sf1:.2%}     {cr:.2%}          {total:>4}")

    # 业务主链路意图（FIND_SHIP, QUERY_SHIP, SAVE_ORDER, QUERY_FREIGHT）
    print("\n【业务主链路意图】")
    main_intents = ["FIND_SHIP", "QUERY_SHIP", "SAVE_ORDER", "QUERY_FREIGHT"]
    for intent in main_intents:
        stats = per_intent_acc.get(intent, {"accuracy": 0, "correct": 0, "total": 0})
        slot_f1_info = per_intent_slot_f1.get(intent, {"f1": 0, "precision": 0, "recall": 0})
        clarify_info = per_intent_clarify_recall.get(intent, {"recall": 0, "total": 0})

        cn_name = INTENT_LABELS_CN.get(intent, intent)
        acc = stats["accuracy"]
        total = stats["total"]
        sf1 = slot_f1_info.get("f1", 0)
        sp = slot_f1_info.get("precision", 0)
        sr = slot_f1_info.get("recall", 0)
        cr = clarify_info.get("recall", 0)
        ct = clarify_info.get("total", 0)

        print(f"  {cn_name}({intent}):")
        print(f"    意图准确率: {acc:.2%} ({stats['correct']}/{total})")
        print(f"    槽位 F1: {sf1:.2%} (P={sp:.2%}, R={sr:.2%})")
        if ct > 0:
            print(f"    Clarify Recall: {cr:.2%} ({clarify_info['correct']}/{ct})")

    bad_cases = results.get("bad_cases", [])
    if bad_cases:
        print(f"\n【Bad Cases】(共 {len(bad_cases)} 条)")
        for i, case in enumerate(bad_cases[:10], 1):
            print(f"\n  #{i} ID: {case.get('id', 'N/A')}")
            print(f"     Query: {case.get('query', '')[:50]}")
            gold = case.get("gold", {})
            pred = case.get("pred", {})
            print(f"     预期: {INTENT_LABELS_CN.get(gold.get('intent',''), gold.get('intent',''))} | 预测: {INTENT_LABELS_CN.get(pred.get('intent',''), pred.get('intent',''))}")

    # 保存 bad cases 到指定目录
    if bad_cases:
        badcase_dir = Path(args.badcase_dir) if args.badcase_dir else (Path(test_path).parent.parent / "data" / "badcase")
        save_badcase_to_dir(bad_cases, badcase_dir, INTENT_LABELS)

    print("\n" + "=" * 60)
    print(f"评测结果已保存到: {output_path}")
    print("=" * 60 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评测结果已保存到: %s", output_path)


if __name__ == "__main__":
    main()
