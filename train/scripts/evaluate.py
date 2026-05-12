"""
BERT 意图识别模型评测脚本

覆盖《04-评估指标设计.md》中的核心离线指标：
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
    per_intent = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, sample in enumerate(samples):
        gold_intent = sample.get("label", {}).get("intent", "UNKNOWN")
        pred_intent = ID_TO_INTENT.get(intent_preds[idx], "UNKNOWN")
        pred_slot_set = {entity_key(e) for e in slot_preds[idx]}
        gold_slot_set = {entity_key(e) for e in slot_golds[idx]}
        intent_ok = pred_intent == gold_intent
        slots_ok = pred_slot_set == gold_slot_set
        clarify_ok = bool(clarify_preds[idx]) == bool(sample.get("label", {}).get("need_clarify", False))

        per_intent[gold_intent]["total"] += 1
        if intent_ok:
            per_intent[gold_intent]["correct"] += 1
        if intent_ok and slots_ok:
            intent_slots_match_count += 1
        if intent_ok and slots_ok and clarify_ok:
            exact_match_count += 1

        if sample.get("label", {}).get("context_inherited"):
            context_total += 1
            if intent_ok and slots_ok:
                context_correct += 1

    total = len(samples)
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
        "per_intent_accuracy": {
            intent: {
                "accuracy": safe_div(stats["correct"], stats["total"]),
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for intent, stats in sorted(per_intent.items())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 意图识别评测脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="", help="评估结果输出路径，默认写到测试集同目录")
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

    logger.info("=" * 50)
    logger.info("意图识别评估结果")
    logger.info("=" * 50)
    logger.info("Intent Accuracy: %.4f", results["intent"]["accuracy"])
    logger.info("Intent Macro F1: %.4f", results["intent"]["f1_macro"])
    logger.info("Intent Weighted F1: %.4f", results["intent"]["f1_weighted"])
    logger.info("Clarify Precision: %.4f", results["clarify"]["precision"])
    logger.info("Clarify Recall: %.4f", results["clarify"]["recall"])
    logger.info("Clarify F1: %.4f", results["clarify"]["f1"])
    logger.info("Slot Entity F1: %.4f", results["slot_entity"]["f1"])
    logger.info("Exact Match: %.4f", results["end_to_end"]["exact_match"])
    logger.info("Context Inheritance Rate: %.4f", results["end_to_end"]["context_inheritance_rate"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评测结果已保存到: %s", output_path)


if __name__ == "__main__":
    main()
