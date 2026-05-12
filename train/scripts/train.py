"""
BERT 意图识别模型训练脚本

模型采用三头联合训练：
1. intent_head：意图分类
2. slot_head：槽位 BIO 序列标注
3. clarify_head：是否需要澄清

用法：
    python -m train.scripts.train \
        --data_dir ./train/data/processed \
        --train_file train.jsonl \
        --valid_file valid.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train.common import (
    BIO_LABELS,
    INTENT_LABELS,
    MODEL_NAME,
    IntentDataset,
    log_validation_result,
    read_jsonl,
    validate_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
INTENT_LOSS_WEIGHT = 1.0
SLOT_LOSS_WEIGHT = 0.5
CLARIFY_LOSS_WEIGHT = 0.5


class BertIntentClassifier(nn.Module):
    """BERT 意图 + 槽位 + 澄清联合分类模型。"""

    def __init__(self, model_name: str, num_intents: int, num_bio_labels: int, num_clarify: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_bio_labels)
        self.clarify_head = nn.Linear(hidden_size, num_clarify)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        # 部分中文 RoBERTa 没有稳定的 pooler_output，直接取 CLS token 更通用。
        cls_output = sequence_output[:, 0, :]
        intent_logits = self.intent_head(cls_output)
        slot_logits = self.slot_head(sequence_output)
        clarify_logits = self.clarify_head(cls_output)
        return intent_logits, slot_logits, clarify_logits


def compute_loss(
    intent_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    clarify_logits: torch.Tensor,
    intent_ids: torch.Tensor,
    bio_labels: torch.Tensor,
    clarify_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """计算三头联合损失，padding 位置不参与槽位损失。"""
    intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_ids)
    clarify_loss = nn.CrossEntropyLoss()(clarify_logits, clarify_ids)

    active_labels = bio_labels.masked_fill(attention_mask == 0, -100)
    slot_loss = nn.CrossEntropyLoss(ignore_index=-100)(
        slot_logits.view(-1, slot_logits.size(-1)),
        active_labels.view(-1),
    )

    return (
        INTENT_LOSS_WEIGHT * intent_loss
        + SLOT_LOSS_WEIGHT * slot_loss
        + CLARIFY_LOSS_WEIGHT * clarify_loss
    )


def train_epoch(model, dataloader, optimizer, scheduler, device: torch.device) -> float:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(dataloader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_ids = batch["intent_id"].to(device)
        bio_labels = batch["bio_labels"].to(device)
        clarify_ids = batch["clarify_id"].to(device)

        optimizer.zero_grad()
        intent_logits, slot_logits, clarify_logits = model(input_ids, attention_mask)
        loss = compute_loss(
            intent_logits,
            slot_logits,
            clarify_logits,
            intent_ids,
            bio_labels,
            clarify_ids,
            attention_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 20 == 0:
            logger.info("step=%s loss=%.4f", step, total_loss / step)

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    """验证集轻量评估，用于选择最佳 checkpoint。"""
    model.eval()
    correct_intent = 0
    correct_clarify = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_ids = batch["intent_id"].to(device)
            bio_labels = batch["bio_labels"].to(device)
            clarify_ids = batch["clarify_id"].to(device)

            intent_logits, slot_logits, clarify_logits = model(input_ids, attention_mask)
            loss = compute_loss(
                intent_logits,
                slot_logits,
                clarify_logits,
                intent_ids,
                bio_labels,
                clarify_ids,
                attention_mask,
            )
            total_loss += loss.item()

            correct_intent += (intent_logits.argmax(dim=-1) == intent_ids).sum().item()
            correct_clarify += (clarify_logits.argmax(dim=-1) == clarify_ids).sum().item()
            total += input_ids.size(0)

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "intent_accuracy": correct_intent / total if total else 0.0,
        "clarify_accuracy": correct_clarify / total if total else 0.0,
    }


def save_checkpoint(
    model: BertIntentClassifier,
    tokenizer,
    output_dir: Path,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """保存模型权重、tokenizer 和训练元信息。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.bin")
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": args.model_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "max_length": args.max_length,
        "intent_labels": INTENT_LABELS,
        "bio_labels": BIO_LABELS,
        "metrics": metrics,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def validate_or_exit(filepath: Path) -> None:
    """训练前先做数据格式校验，错误直接停止，warning 只提示。"""
    samples = read_jsonl(filepath)
    result = validate_samples(samples, source_name=str(filepath))
    log_validation_result(result)
    if not result.ok:
        logger.error("数据校验失败，请先修复上面的错误。")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 意图识别训练脚本")
    parser.add_argument("--data_dir", type=str, default="./train/data/processed", help="数据目录")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="训练集文件名")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl", help="验证集文件名")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HuggingFace 预训练模型")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--output_dir", type=str, default="./train/outputs/checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info(
        "训练配置: model=%s epoch=%s batch_size=%s max_length=%s device=%s",
        args.model_name,
        args.max_epochs,
        args.batch_size,
        args.max_length,
        args.device,
    )

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    valid_path = data_dir / args.valid_file

    if not train_path.exists():
        logger.error("训练文件不存在: %s", train_path)
        sys.exit(1)

    logger.info("校验训练集...")
    validate_or_exit(train_path)
    if valid_path.exists():
        logger.info("校验验证集...")
        validate_or_exit(valid_path)
    else:
        logger.warning("验证集不存在: %s，本次只保存最后一轮模型。", valid_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        logger.warning("当前 tokenizer 不是 fast tokenizer，槽位 offset 对齐可能不可用。")

    train_dataset = IntentDataset(train_path, tokenizer, max_length=args.max_length)
    valid_dataset = IntentDataset(valid_path, tokenizer, max_length=args.max_length) if valid_path.exists() else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size) if valid_dataset else None

    device = torch.device(args.device)
    model = BertIntentClassifier(
        model_name=args.model_name,
        num_intents=len(INTENT_LABELS),
        num_bio_labels=len(BIO_LABELS),
    ).to(device)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = -1.0
    output_dir = Path(args.output_dir)

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info("Epoch %s/%s - train_loss=%.4f", epoch, args.max_epochs, train_loss)

        if valid_loader:
            metrics = evaluate(model, valid_loader, device)
            score = metrics["intent_accuracy"] + 0.2 * metrics["clarify_accuracy"] - 0.1 * metrics["loss"]
            logger.info(
                "Epoch %s - valid_loss=%.4f intent_acc=%.4f clarify_acc=%.4f score=%.4f",
                epoch,
                metrics["loss"],
                metrics["intent_accuracy"],
                metrics["clarify_accuracy"],
                score,
            )
            if score > best_score:
                best_score = score
                save_checkpoint(model, tokenizer, output_dir, metrics, args)
                logger.info("保存最优模型到 %s", output_dir / "best_model.bin")
        else:
            save_checkpoint(model, tokenizer, output_dir, {"train_loss": train_loss}, args)

    logger.info("训练完成，模型目录: %s", output_dir)


if __name__ == "__main__":
    main()
