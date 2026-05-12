"""
意图识别训练公共模块

这里集中放置标签体系、JSONL 数据校验、输入拼接和 BIO 对齐逻辑。
训练脚本、评估脚本、后续推理服务都应尽量复用本模块，避免“训练时一套、
评估时一套、线上又一套”的隐性偏差。
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_HISTORY_TURNS = 3

INTENT_LABELS = [
    "DOC_QA",
    "FIND_SHIP",
    "SAVE_ORDER",
    "QUERY_ORDER",
    "QUERY_SHIP",
    "QUERY_FREIGHT",
    "QUERY_WEATHER",
    "QUERY_WATER_LEVEL",
    "DISPATCH_MONITOR",
    "IMAGE_OCR",
    "FEEDBACK",
    "QUERY_OIL_STATION",
    "QUERY_SHIP_INFO",
    "TALK",
]

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

INTENT_TO_ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID_TO_INTENT = {idx: label for label, idx in INTENT_TO_ID.items()}

# 槽位使用标准 BIO 标签：O + 每个槽位的 B-/I-。
BIO_LABELS = ["O"]
for slot_name in SLOT_LABELS:
    BIO_LABELS.extend([f"B-{slot_name}", f"I-{slot_name}"])
BIO_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_BIO = {idx: label for label, idx in BIO_TO_ID.items()}


@dataclass
class ValidationResult:
    """JSONL 校验结果，便于脚本和测试复用。"""

    total: int
    errors: list[str]
    warnings: list[str]
    intent_counts: dict[str, int]
    slot_counts: dict[str, int]
    clarify_count: int

    @property
    def ok(self) -> bool:
        return not self.errors


def read_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，保留原始样本结构。"""
    samples: list[dict[str, Any]] = []
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
    return samples


def write_jsonl(samples: list[dict[str, Any]], filepath: str | Path) -> None:
    """写入 JSONL 文件。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_model_text(sample: dict[str, Any], max_history_turns: int = MAX_HISTORY_TURNS) -> str:
    """按训练方案拼接历史上下文和当前问题。"""
    parts: list[str] = []
    for turn in sample.get("history", [])[-max_history_turns:]:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"[H_USER] {content}")
        elif role == "assistant":
            parts.append(f"[H_ASSISTANT] {content}")
        else:
            parts.append(f"[H_{role.upper() or 'UNKNOWN'}] {content}")

    query = str(sample.get("query", "")).strip()
    parts.append(f"[QUERY] {query}")
    return "\n".join(parts)


def normalize_slot_values(value: Any) -> list[str]:
    """槽位值既支持字符串，也兼容列表；空值会被过滤。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def find_slot_spans(text: str, slots: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    """在拼接后的输入文本中定位槽位实体。

    返回值为 (start, end, slot_name, slot_value)。如果同一个实体出现多次，只标第一个
    未重叠位置，避免历史和 query 中重复实体导致监督信号过密。
    """
    spans: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    # 长实体优先，避免“南京港”被“南京”抢先占位。
    items: list[tuple[str, str]] = []
    for slot_name, raw_value in slots.items():
        if slot_name not in SLOT_LABELS:
            continue
        for value in normalize_slot_values(raw_value):
            items.append((slot_name, value))
    items.sort(key=lambda item: len(item[1]), reverse=True)

    for slot_name, value in items:
        start = text.find(value)
        while start >= 0:
            end = start + len(value)
            has_overlap = any(not (end <= s or start >= e) for s, e in occupied)
            if not has_overlap:
                spans.append((start, end, slot_name, value))
                occupied.append((start, end))
                break
            start = text.find(value, start + 1)

    return spans


def align_bio_labels(
    text: str,
    slots: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> torch.Tensor:
    """使用 tokenizer 的 offset_mapping 将字符级实体对齐到 token 级 BIO 标签。"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    labels = [BIO_TO_ID["O"]] * len(offsets)

    spans = find_slot_spans(text, slots)
    for start, end, slot_name, _ in spans:
        token_indices: list[int] = []
        for idx, (token_start, token_end) in enumerate(offsets):
            # 特殊 token 和 padding 的 offset 通常是 (0, 0)，需要跳过。
            if token_start == token_end:
                continue
            if token_end <= start or token_start >= end:
                continue
            token_indices.append(idx)

        for pos, token_idx in enumerate(token_indices):
            prefix = "B" if pos == 0 else "I"
            labels[token_idx] = BIO_TO_ID[f"{prefix}-{slot_name}"]

    return torch.tensor(labels[:max_length], dtype=torch.long)


def validate_samples(samples: list[dict[str, Any]], source_name: str = "<memory>") -> ValidationResult:
    """校验样本格式、标签合法性，并统计标签分布。"""
    errors: list[str] = []
    warnings: list[str] = []
    intent_counter: Counter[str] = Counter()
    slot_counter: Counter[str] = Counter()
    clarify_count = 0
    seen_ids: set[str] = set()

    for idx, sample in enumerate(samples, 1):
        prefix = f"{source_name}:{idx}"
        for field in ("id", "history", "query", "label"):
            if field not in sample:
                errors.append(f"{prefix} 缺少必填字段 {field}")

        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            errors.append(f"{prefix} id 不能为空")
        elif sample_id in seen_ids:
            warnings.append(f"{prefix} id 重复: {sample_id}")
        seen_ids.add(sample_id)

        if not isinstance(sample.get("history", []), list):
            errors.append(f"{prefix} history 必须是数组")
        else:
            for turn_idx, turn in enumerate(sample.get("history", []), 1):
                role = turn.get("role") if isinstance(turn, dict) else None
                content = turn.get("content") if isinstance(turn, dict) else None
                if role not in {"user", "assistant"}:
                    warnings.append(f"{prefix} history[{turn_idx}] role 非标准值: {role}")
                if not isinstance(content, str) or not content.strip():
                    warnings.append(f"{prefix} history[{turn_idx}] content 为空")

        if not str(sample.get("query", "")).strip():
            errors.append(f"{prefix} query 不能为空")

        label = sample.get("label", {})
        if not isinstance(label, dict):
            errors.append(f"{prefix} label 必须是对象")
            continue

        intent = label.get("intent")
        if intent not in INTENT_TO_ID:
            errors.append(f"{prefix} 非法 intent: {intent}")
        else:
            intent_counter[intent] += 1

        slots = label.get("slots", {})
        if not isinstance(slots, dict):
            errors.append(f"{prefix} label.slots 必须是对象")
        else:
            text = build_model_text(sample)
            for slot_name, raw_value in slots.items():
                if slot_name not in SLOT_LABELS:
                    errors.append(f"{prefix} 非法 slot: {slot_name}")
                    continue
                values = normalize_slot_values(raw_value)
                if not values:
                    warnings.append(f"{prefix} slot {slot_name} 值为空")
                for value in values:
                    slot_counter[slot_name] += 1
                    if value not in text:
                        warnings.append(f"{prefix} slot {slot_name}={value} 未出现在输入文本中")

        need_clarify = label.get("need_clarify")
        if not isinstance(need_clarify, bool):
            errors.append(f"{prefix} label.need_clarify 必须是 bool")
        elif need_clarify:
            clarify_count += 1
            if not str(label.get("clarify_question", "")).strip():
                warnings.append(f"{prefix} need_clarify=true 但未填写 clarify_question")

    missing_intents = [intent for intent in INTENT_LABELS if intent not in intent_counter]
    if missing_intents:
        warnings.append(f"{source_name} 缺少 intent 样本: {', '.join(missing_intents)}")

    return ValidationResult(
        total=len(samples),
        errors=errors,
        warnings=warnings,
        intent_counts=dict(intent_counter),
        slot_counts=dict(slot_counter),
        clarify_count=clarify_count,
    )


def log_validation_result(result: ValidationResult) -> None:
    """用中文日志打印校验摘要。"""
    logger.info("样本总数: %s", result.total)
    logger.info("意图分布: %s", result.intent_counts)
    logger.info("槽位分布: %s", result.slot_counts)
    logger.info("澄清样本数: %s", result.clarify_count)
    for warning in result.warnings[:20]:
        logger.warning(warning)
    if len(result.warnings) > 20:
        logger.warning("其余 warning 已省略: %s 条", len(result.warnings) - 20)
    for error in result.errors:
        logger.error(error)


class IntentDataset(Dataset):
    """JSONL 数据集加载器，输出联合训练所需的 tensor。"""

    def __init__(self, filepath: str | Path, tokenizer: Any, max_length: int = 256):
        self.filepath = Path(filepath)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = read_jsonl(self.filepath)
        logger.info("加载样本 %s 条 from %s", len(self.samples), self.filepath)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = build_model_text(sample)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = sample.get("label", {})
        intent = label.get("intent", "TALK")
        intent_id = INTENT_TO_ID.get(intent, INTENT_TO_ID["TALK"])
        clarify_id = 1 if label.get("need_clarify", False) else 0
        bio_labels = align_bio_labels(
            text=text,
            slots=label.get("slots", {}),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_id": torch.tensor(intent_id, dtype=torch.long),
            "clarify_id": torch.tensor(clarify_id, dtype=torch.long),
            "bio_labels": bio_labels,
        }


def decode_bio_entities(
    bio_ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
) -> list[dict[str, Any]]:
    """将模型预测的 BIO 序列还原为实体列表。"""
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close_current() -> None:
        nonlocal current
        if current is None:
            return
        current["text"] = text[current["start"] : current["end"]]
        entities.append(current)
        current = None

    for bio_id, (start, end) in zip(bio_ids, offsets):
        if start == end:
            close_current()
            continue

        label = ID_TO_BIO.get(int(bio_id), "O")
        if label == "O":
            close_current()
            continue

        prefix, slot_name = label.split("-", 1)
        if prefix == "B" or current is None or current["slot"] != slot_name:
            close_current()
            current = {"slot": slot_name, "start": start, "end": end}
        else:
            current["end"] = end

    close_current()
    return entities


def gold_slot_entities(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """从标注 slots 中构建实体列表，供 Entity F1 和 Exact Match 使用。"""
    text = build_model_text(sample)
    entities: list[dict[str, Any]] = []
    for start, end, slot_name, value in find_slot_spans(text, sample.get("label", {}).get("slots", {})):
        entities.append({"slot": slot_name, "text": value, "start": start, "end": end})
    return entities


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    """实体级评估只要求槽位类型和完整文本一致，不强依赖字符位置。"""
    return str(entity["slot"]), str(entity["text"])


def confusion_matrix_counts(labels: list[int], preds: list[int]) -> dict[str, int]:
    """二分类统计，Clarify Precision/Recall/F1 使用。"""
    counts = defaultdict(int)
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            counts["tp"] += 1
        elif label == 0 and pred == 1:
            counts["fp"] += 1
        elif label == 1 and pred == 0:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return dict(counts)


def safe_div(numerator: float, denominator: float) -> float:
    """避免指标计算时除零。"""
    return numerator / denominator if denominator else 0.0
