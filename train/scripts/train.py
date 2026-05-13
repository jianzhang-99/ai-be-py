"""
BERT 意图识别模型训练脚本

模型采用三头联合训练：
1. intent_head：意图分类（多分类，如"查运价"、"查船期"等）
2. slot_head：槽位 BIO 序列标注（用于实体抽取，如货物名、目的港）
3. clarify_head：是否需要澄清（二分类，判断当前对话是否需要追问）

三头联合训练的优势：
- 共享 BERT encoder，减少参数量
- 意图和槽位任务互补，提升整体理解能力
- clarify 任务辅助对话策略决策

损失加权策略：
- intent_loss_weight=1.0：主任务，全量加权
- slot_loss_weight=1.0：辅助任务，提升权重解决槽位抽取全0问题
- clarify_loss_weight=2.0：辅助任务，大幅提升权重配合FocalLoss解决澄清判断全0问题
- clarify使用FocalLoss(gamma=2.0, pos_weight=10.0)处理极端不平衡

模型选择策略：
- score = intent_accuracy + 0.5 * clarify_accuracy - 0.05 * loss
- 优先保证clarify准确率，避免模型选择偏向意图而忽略澄清

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

# 延迟导入避免循环依赖
_import_err = None
try:
    from train.scripts.evaluate import evaluate_model
except Exception as e:
    _import_err = str(e)
    evaluate_model = None

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
SLOT_LOSS_WEIGHT = 1.0  # 第一阶段用 1.0，后续会显著提高
CLARIFY_LOSS_WEIGHT = 2.0  # 大幅提升澄清权重，解决clarify全0问题
VERSIONS_FILE = "versions.json"

# Focal Loss Gamma参数，用于处理类别不平衡
FOCAL_GAMMA = 2.0
# Clarify正负样本权重比（正样本约2%，需要大幅提升）
CLARIFY_POS_WEIGHT = 10.0

# ==================== 分阶段训练配置 ====================
# 第一阶段：训 intent + clarify，拿到稳定语义底座
STAGE1_EPOCHS = 3  # 第一阶段轮数
STAGE1_INTENT_WEIGHT = 1.0
STAGE1_CLARIFY_WEIGHT = 2.0
STAGE1_SLOT_WEIGHT = 0.5  # 槽位权重放低，让模型先学语义

# 第二阶段：降低 intent 权重，大幅提高 slot 权重，加 O 类降权
STAGE2_EPOCHS = 4  # 第二阶段轮数
STAGE2_INTENT_WEIGHT = 0.5
STAGE2_CLARIFY_WEIGHT = 1.0
STAGE2_SLOT_WEIGHT = 4.0  # 槽位权重从 1.0 提到 4.0~5.0

# ==================== BIO 标签类别权重 ====================
# O 类占比太高，需要降权，否则模型容易学成全 O
# BIO_LABELS = ["O"] + ["B-{slot}", "I-{slot}"] for each slot
BIO_O_WEIGHT = 0.3  # O 类权重，降低避免模型倾向全 O
BIO_ENTITY_WEIGHT = 1.0  # 实体标签正常权重

# 构建 BIO 类别权重列表（用于 slot loss 计算）
from train.common import SLOT_LABELS as _SLOT_LABELS
BIO_CLASS_WEIGHTS: list[float] = [BIO_O_WEIGHT]  # O 类
for _ in _SLOT_LABELS:
    BIO_CLASS_WEIGHTS.append(BIO_ENTITY_WEIGHT)  # B-{slot}
    BIO_CLASS_WEIGHTS.append(BIO_ENTITY_WEIGHT)  # I-{slot}


def get_next_version(outputs_dir: Path) -> str:
    """自动获取下一个版本号 Vx.x"""
    versions_file = outputs_dir / VERSIONS_FILE
    if versions_file.exists():
        with versions_file.open("r", encoding="utf-8") as f:
            versions = json.load(f)
    else:
        versions = []

    if not versions:
        return "V0.1"

    # 解析已有版本号，找出最大值
    max_major = 0
    max_minor = 0
    for v in versions:
        if v.startswith("V") and "." in v:
            try:
                parts = v[1:].split(".")
                major = int(parts[0])
                minor = int(parts[1])
                if major > max_major or (major == max_major and minor > max_minor):
                    max_major = major
                    max_minor = minor
            except ValueError:
                continue

    # 递增
    if max_minor >= 9:
        max_major += 1
        max_minor = 0
    else:
        max_minor += 1

    return f"V{max_major}.{max_minor}"


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    特别适用于 clarify 任务这种极端不平衡场景（正样本约2%）。
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 使用 weight 参数代替 pos_weight，处理类别不平衡
        weight = torch.tensor([1.0, self.pos_weight], device=logits.device)
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none", weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class CRF(nn.Module):
    """
    条件随机场层（用于槽位 BIO 序列标注）

    CRF 相比纯 argmax 的优势：
    - 能够建模标签之间的转移约束（如 B-X 后面应该是 I-X 或 O，不应该是 B-Y）
    - 适合 BIO 序列约束
    """

    def __init__(self, num_labels: int, init_transitions: float = -2.0):
        super().__init__()
        self.num_labels = num_labels
        # 转移矩阵：from_tag -> to_tag，初始值倾向"B后面是I"和"I后面是I"
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.1)
        # 初始化：对角线倾向于保持标签，O->O 更常见
        with torch.no_grad():
            for i in range(num_labels):
                self.transitions[i, i].fill_(-1.0)  # 倾向于跳到自己（O->O, I->I）
            # B->I 应该容易（同一实体继续）
            for i in range(1, num_labels):
                if i % 2 == 1:  # B-{slot}
                    b_idx = i
                    i_idx = i + 1
                    if i_idx < num_labels:
                        self.transitions[b_idx, i_idx].fill_(-0.5)
                        self.transitions[i_idx, i_idx].fill_(-0.3)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算负对数似然损失。

        参数：
            emissions: 发射分数，(batch, seq_len, num_labels)
            tags: 真实标签，(batch, seq_len)
            mask: 有效位置掩码，(batch, seq_len)

        返回：
            负对数似然损失，标量张量
        """
        batch_size, seq_len = tags.shape
        mask = mask.float()

        # 计算路径分数：发射分数 + 转移分数
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return (partition - score).mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算一条路径的分数：sum(emission) + sum(transition)"""
        seq_len = emissions.size(1)

        # 发射分数：取每个位置的标签对应的发射值
        # tags[:, 1:] 避开第一个位置（没有转移来源）
        # transitions[tags[:, :-1], tags[:, 1:]] 是转移分数
        r = torch.arange(batch_size).unsqueeze(1)
        # emission at each step
        score = emissions[r, torch.arange(seq_len), tags].sum(dim=1)

        # 转移分数：从第1个位置开始（i-1 到 i）
        for i in range(1, seq_len):
            valid = mask[:, i].float()
            trans_score = self.transitions[tags[:, i-1], tags[:, i]]
            score = score + trans_score * valid

        return score

    def _compute_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向算法计算配分函数（log-sum-exp）"""
        seq_len = emissions.size(1)
        batch_size = emissions.size(0)
        history: list[torch.Tensor] = []

        for i in range(seq_len):
            emit_score = emissions[:, i, :]  # (batch, num_labels)
            mask_val = mask[:, i].float()

            if i == 0:
                current = emit_score
            else:
                # 上一个位置的分数（考虑mask）
                prev = history[-1]  # (batch, num_labels)
                # prev[:, None, :] + self.transitions: (batch, num_labels, num_labels)
                # + emit_score[:, None, :]: (batch, num_labels, num_labels)
                prev = prev.unsqueeze(2)  # (batch, num_labels, 1)
                trans = self.transitions.unsqueeze(0)  # (1, num_labels, num_labels)
                scores = prev + trans  # (batch, num_labels, num_labels)
                scores = scores + emit_score.unsqueeze(1)  # (batch, num_labels, num_labels)

                # mask：当前有效位置取所有标签，之前的位置保持原样
                mask_val = mask_val.view(batch_size, 1, 1)
                scores = scores * mask_val + (-10000.0) * (1 - mask_val)

                # log-sum-exp over last dimension
                current = torch.logsumexp(scores, dim=2)  # (batch, num_labels)

            history.append(current)

        # 最后一个位置
        return torch.logsumexp(history[-1], dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """维特比解码，贪心解码简化版（适合 O 类占比高的情况）"""
        seq_len = emissions.size(1)
        batch_size = emissions.size(0)
        mask_np = mask.cpu().numpy()

        results: list[list[int]] = []

        # 贪心解码：每步取最大分数的标签，同时考虑转移约束
        # 简化实现：只考虑从 O 不能转到 B（应该是 I 或 O）
        for b in range(batch_size):
            path: list[int] = []
            prev_tag = 0  # 默认从 O 开始

            for i in range(seq_len):
                if not mask_np[b, i]:
                    break

                # 发射分数
                emit = emissions[b, i].cpu().numpy()

                # 贪心选择：发射 + 转移
                best_score = -float('inf')
                best_tag = 0

                for tag in range(self.num_labels):
                    score = emit[tag]
                    # 转移惩罚
                    if i > 0:
                        score += self.transitions[prev_tag, tag].item()
                    # B 后面不能直接跟 O（实体被截断）
                    if prev_tag != 0 and prev_tag % 2 == 1 and tag == 0:  # B-X -> O
                        score -= 2.0  # 惩罚
                    if score > best_score:
                        best_score = score
                        best_tag = tag

                path.append(best_tag)
                prev_tag = best_tag

            results.append(path)

        return results


class BertIntentClassifier(nn.Module):
    """
    BERT 意图 + 槽位 + 澄清联合分类模型。

    结构：
        BERT Encoder（共享）→ Dropout → 三路独立分类头

    三路分类头：
        - intent_head：意图分类，输出 (batch, num_intents)
        - slot_head：槽位 BIO 序列标注，输出 (batch, seq_len, num_bio_labels)
        - clarify_head：澄清判断，输出 (batch, num_clarify)

    为什么用 CLS token 做意图/澄清，而用完整序列做槽位：
        - CLS token 经过 BERT 的pooler或直接取[0]位，聚合了整句信息，适合分类
        - 槽位标注需要每个 token 的表示，所以用 last_hidden_state 完整序列
    """

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
    intent_weight: float = INTENT_LOSS_WEIGHT,
    slot_weight: float = SLOT_LOSS_WEIGHT,
    clarify_weight: float = CLARIFY_LOSS_WEIGHT,
    use_crf: bool = False,
    crf_layer: CRF = None,
) -> torch.Tensor:
    """
    计算三头联合损失。

    参数：
        intent_logits：意图 logits，(batch, num_intents)
        slot_logits：槽位 logits，(batch, seq_len, num_bio_labels)
        clarify_logits：澄清 logits，(batch, num_clarify)
        intent_ids：意图标签，(batch,)
        bio_labels：槽位标签，(batch, seq_len)
        clarify_ids：澄清标签，(batch,)
        attention_mask：注意力掩码，(batch, seq_len)
        intent_weight：意图损失权重
        slot_weight：槽位损失权重
        clarify_weight：澄清损失权重
        use_crf：是否使用 CRF 层
        crf_layer：CRF 层实例

    返回：
        加权求和后的总损失，标量张量

    注意：
        - intent_loss 使用标准 CrossEntropyLoss
        - clarify_loss 使用 FocalLoss 处理极度不平衡（正样本约2%）
        - slot_loss 使用带 BIO 类别权重的 CrossEntropyLoss，O 类降权
        - 最终损失 = intent_loss*intent_weight + slot_loss*slot_weight + clarify_loss*clarify_weight
    """
    intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_ids)
    # 使用FocalLoss处理clarify的极度不平衡问题
    clarify_loss_fn = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=CLARIFY_POS_WEIGHT)
    clarify_loss = clarify_loss_fn(clarify_logits, clarify_ids)

    # 获取有效标签（排除 padding）
    active_labels = bio_labels.masked_fill(attention_mask == 0, -100)
    active_mask = (attention_mask == 1)

    if use_crf and crf_layer is not None:
        # 使用 CRF 层计算损失
        slot_loss = crf_layer(emissions=slot_logits, tags=bio_labels, mask=active_mask)
    else:
        # 使用带类别权重的 CrossEntropyLoss，O 类降权
        weight = torch.tensor(BIO_CLASS_WEIGHTS, device=bio_labels.device)
        slot_loss = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)(
            slot_logits.view(-1, slot_logits.size(-1)),
            active_labels.view(-1),
        )

    return (
        intent_weight * intent_loss
        + slot_weight * slot_loss
        + clarify_weight * clarify_loss
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    intent_weight: float = INTENT_LOSS_WEIGHT,
    slot_weight: float = SLOT_LOSS_WEIGHT,
    clarify_weight: float = CLARIFY_LOSS_WEIGHT,
    use_crf: bool = False,
    crf_layer: CRF = None,
) -> float:
    """
    训练一个 epoch。

    流程：
        1. 遍历 dataloader 获取 batch 数据
        2. 前向传播获取三路 logits
        3. 计算联合损失（带分阶段权重）
        4. 反向传播 + 梯度裁剪 + 参数更新
        5. 学习率 scheduler 更新

    参数：
        model：BERT 联合分类模型
        dataloader：训练数据加载器
        optimizer：AdamW 优化器
        scheduler：学习率调度器
        device：计算设备（cuda/cpu）
        intent_weight：意图损失权重（分阶段调整）
        slot_weight：槽位损失权重（分阶段调整）
        clarify_weight：澄清损失权重（分阶段调整）
        use_crf：是否使用 CRF 层
        crf_layer：CRF 层实例

    返回：
        本 epoch 的平均损失
    """
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
            intent_weight=intent_weight,
            slot_weight=slot_weight,
            clarify_weight=clarify_weight,
            use_crf=use_crf,
            crf_layer=crf_layer,
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
    """
    验证集评估。

    计算三个指标：
        - loss：联合损失在验证集上的均值
        - intent_accuracy：意图分类准确率
        - clarify_accuracy：澄清判断准确率

    注意：
        - 不计算 slot_accuracy（槽位标注通常需要更复杂的评估方式，如 F1）
        - 模型切换到 eval 模式，禁用 dropout 和 batch normalization 更新

    返回：
        包含 loss、intent_accuracy、clarify_accuracy 的字典
    """
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
    version: str = "",
) -> None:
    """
    保存模型检查点。

    保存内容：
        - best_model.bin：模型权重（state_dict）
        - tokenizer/：HuggingFace tokenizer 文件
        - metadata.json：训练元信息（模型名、超参数、标签集、指标）

    注意：
        - 只保存权重，不保存优化器状态（轻量化保存）
        - tokenizer 用 save_pretrained 保存，load 时也用 from_pretrained 恢复
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.bin")
    tokenizer.save_pretrained(output_dir)

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

    metadata = {
        "版本": version,
        "模型名称": args.model_name,
        "保存时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最大序列长度": args.max_length,
        "意图标签": [INTENT_LABELS_CN.get(l, l) for l in INTENT_LABELS],
        "槽位标签": [l for l in BIO_LABELS],
        "训练参数": {
            "epoch数": args.max_epochs,
            "batch_size": args.batch_size,
            "学习率": args.lr,
        },
        "验证指标": {
            "loss": round(metrics.get("loss", 0), 4),
            "意图准确率": round(metrics.get("intent_accuracy", 0), 4),
            "澄清准确率": round(metrics.get("clarify_accuracy", 0), 4),
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def validate_or_exit(filepath: Path) -> None:
    """
    训练前数据格式校验，错误直接退出。

    校验内容（参见 validate_samples）：
        - 必填字段是否存在（text, intent, slots, clarify）
        - intent 标签是否在预定义标签集中
        - slots BIO 标注格式是否合法
        - clarify 值是否为 0 或 1

    注意：
        - warning 不阻止训练（数据可能不完美但可跑通）
        - error 会触发 sys.exit(1)，必须先修复
    """
    samples = read_jsonl(filepath)
    result = validate_samples(samples, source_name=str(filepath))
    log_validation_result(result)
    if not result.ok:
        logger.error("数据校验失败，请先修复上面的错误。")
        sys.exit(1)


def main() -> None:
    """
    BERT 意图识别模型训练入口。

    完整训练流程：
        1. 参数解析与配置校验
        2. 数据文件存在性检查 + 格式校验
        3. 加载 tokenizer 和数据集
        4. 初始化模型（BERT Encoder + 三路分类头）
        5. 配置优化器和学习率调度器（AdamW + Linear Warmup）
        6. 迭代训练每个 epoch：
            - 训练阶段：前向 → 计算损失 → 反向 → 更新参数
            - 验证阶段（如有）：评估并计算综合得分
            - 保存最优模型（综合得分最高）
        7. 训练完成，输出模型目录

    验证集不存在时的处理：
        - 只在最后一个 epoch 保存模型
        - 无法选择"最优"checkpoint，按需自行判断

    模型选择策略（score 计算）：
        score = intent_accuracy + 0.2 * clarify_accuracy - 0.1 * loss
        意图准确率是主指标，澄清准确率加权0.2，损失作为惩罚项加权0.1
    """
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="BERT 意图识别训练脚本")
    parser.add_argument("--data_dir", type=str, default="./train/data/processed", help="数据目录")
    parser.add_argument("--train_file", type=str, default="all.jsonl", help="训练集文件名")
    parser.add_argument("--valid_file", type=str, default="valid_cleaned.jsonl", help="验证集文件名")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HuggingFace 预训练模型")
    parser.add_argument("--max_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="批大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="最大序列长度")
    parser.add_argument("--output_dir", type=str, default="./train/outputs/checkpoints", help="模型输出目录")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备，优先使用 CUDA"
    )
    parser.add_argument(
        "--eval_after_training", action="store_true", default=True,
        help="训练完成后自动在测试集上评测"
    )
    parser.add_argument(
        "--use_crf", action="store_true", default=False,
        help="第二阶段使用 CRF 层（需要额外训练 CRF 参数）"
    )
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

    # 创建 CRF 层（可选，用于第二阶段）
    crf_layer = CRF(num_labels=len(BIO_LABELS)).to(device)
    use_crf = args.use_crf if hasattr(args, 'use_crf') else False

    # ==================== 两阶段训练 ====================
    # 第一阶段：训 intent + clarify，拿到稳定语义底座
    # 第二阶段：降低 intent 权重，大幅提高 slot 权重，加 O 类降权

    logger.info("=" * 60)
    logger.info("第一阶段训练：intent + clarify（低 slot 权重）")
    logger.info("配置: intent_weight=%.1f, slot_weight=%.1f, clarify_weight=%.1f",
                STAGE1_INTENT_WEIGHT, STAGE1_SLOT_WEIGHT, STAGE1_CLARIFY_WEIGHT)
    logger.info("=" * 60)

    stage1_steps = len(train_loader) * STAGE1_EPOCHS
    stage1_warmup = int(stage1_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, stage1_warmup, stage1_steps)

    best_score = -1.0
    outputs_base = Path(args.output_dir).parent  # ./train/outputs
    version = get_next_version(outputs_base)
    version_dir = outputs_base / version
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("本次训练版本: %s", version)
    logger.info("模型保存目录: %s", checkpoint_dir)

    # 第一阶段训练
    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            intent_weight=STAGE1_INTENT_WEIGHT,
            slot_weight=STAGE1_SLOT_WEIGHT,
            clarify_weight=STAGE1_CLARIFY_WEIGHT,
            use_crf=False,
        )
        logger.info("Stage1 Epoch %s/%s - train_loss=%.4f", epoch, STAGE1_EPOCHS, train_loss)

        if valid_loader:
            metrics = evaluate(model, valid_loader, device)
            score = metrics["intent_accuracy"] + 0.5 * metrics["clarify_accuracy"] - 0.05 * metrics["loss"]
            logger.info(
                "Stage1 Epoch %s - valid_loss=%.4f intent_acc=%.4f clarify_acc=%.4f score=%.4f",
                epoch,
                metrics["loss"],
                metrics["intent_accuracy"],
                metrics["clarify_accuracy"],
                score,
            )
            if score > best_score:
                best_score = score
                save_checkpoint(model, tokenizer, checkpoint_dir, metrics, args, version)
                logger.info("保存最优模型到 %s", checkpoint_dir / "best_model.bin")

    # 第二阶段训练：大幅提高 slot 权重，使用 CRF（可选）和 O 类降权
    logger.info("=" * 60)
    logger.info("第二阶段训练：slot 权重提升 + O 类降权%s", " + CRF" if use_crf else "")
    logger.info("配置: intent_weight=%.1f, slot_weight=%.1f, clarify_weight=%.1f",
                STAGE2_INTENT_WEIGHT, STAGE2_SLOT_WEIGHT, STAGE2_CLARIFY_WEIGHT)
    logger.info("=" * 60)

    stage2_total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    stage2_steps = len(train_loader) * STAGE2_EPOCHS
    stage2_warmup = int(stage2_steps * WARMUP_RATIO)

    # 重新初始化优化器（可选，保持学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=WEIGHT_DECAY)  # 降低学习率
    scheduler = get_linear_schedule_with_warmup(optimizer, stage2_warmup, stage2_steps)

    for epoch in range(STAGE1_EPOCHS + 1, stage2_total_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            intent_weight=STAGE2_INTENT_WEIGHT,
            slot_weight=STAGE2_SLOT_WEIGHT,
            clarify_weight=STAGE2_CLARIFY_WEIGHT,
            use_crf=use_crf,
            crf_layer=crf_layer if use_crf else None,
        )
        logger.info("Stage2 Epoch %s/%s - train_loss=%.4f", epoch, stage2_total_epochs, train_loss)

        if valid_loader:
            metrics = evaluate(model, valid_loader, device)
            # 新评分标准：业务导向
            # score = 0.35 * intent_f1_macro + 0.25 * clarify_f1 + 0.25 * slot_f1 + 0.15 * exact_match
            score = metrics["intent_accuracy"] + 0.25 * metrics["clarify_accuracy"] - 0.05 * metrics["loss"]
            logger.info(
                "Stage2 Epoch %s - valid_loss=%.4f intent_acc=%.4f clarify_acc=%.4f score=%.4f",
                epoch,
                metrics["loss"],
                metrics["intent_accuracy"],
                metrics["clarify_accuracy"],
                score,
            )
            if score > best_score:
                best_score = score
                save_checkpoint(model, tokenizer, checkpoint_dir, metrics, args, version)
                logger.info("保存最优模型到 %s", checkpoint_dir / "best_model.bin")

    logger.info("训练完成，模型目录: %s", checkpoint_dir)

    # 训练完成后自动评测
    if args.eval_after_training and evaluate_model is not None:
        test_path = data_dir / "test_cleaned.jsonl"
        if test_path.exists():
            logger.info("=" * 50)
            logger.info("开始自动评测...")
            logger.info("=" * 50)
            try:
                from torch.utils.data import DataLoader as DL
                test_dataset = IntentDataset(test_path, tokenizer, max_length=args.max_length)
                test_loader = DL(test_dataset, batch_size=args.batch_size)
                samples = read_jsonl(test_path)

                results = evaluate_model(model, test_loader, samples, tokenizer, device)

                # 中文输出
                intent_m = results["intent"]
                clarify_m = results["clarify"]
                slot_m = results["slot_entity"]
                end_to_end = results["end_to_end"]

                print("\n" + "=" * 60)
                print("意图识别模型评测报告")
                print("=" * 60)

                print("\n【意图识别】")
                print(f"  准确率 (Accuracy): {intent_m['accuracy']:.4f}")
                print(f"  Macro F1:          {intent_m['f1_macro']:.4f}")
                print(f"  Weighted F1:       {intent_m['f1_weighted']:.4f}")

                print("\n【澄清判断】")
                print(f"  Precision: {clarify_m['precision']:.4f}")
                print(f"  Recall:    {clarify_m['recall']:.4f}")
                print(f"  F1:        {clarify_m['f1']:.4f}")

                print("\n【槽位抽取】")
                print(f"  Entity F1:    {slot_m['f1']:.4f}")
                print(f"  Entity Recall: {slot_m['recall']:.4f}")
                print(f"  Entity Prec:  {slot_m['precision']:.4f}")

                print("\n【端到端匹配】")
                print(f"  Exact Match:         {end_to_end['exact_match']:.4f}")
                print(f"  Intent+Slots Match:  {end_to_end['intent_slots_match']:.4f}")

                INTENT_LABELS_CN = {
                    "DOC_QA": "文档问答", "FIND_SHIP": "找船", "SAVE_ORDER": "发布运单",
                    "QUERY_ORDER": "查询订单", "QUERY_SHIP": "查船位置", "QUERY_FREIGHT": "运价查询",
                    "QUERY_WEATHER": "天气查询", "QUERY_WATER_LEVEL": "水位查询",
                    "DISPATCH_MONITOR": "在途监控", "IMAGE_OCR": "图片识别",
                    "FEEDBACK": "反馈投诉", "QUERY_OIL_STATION": "加油站查询",
                    "QUERY_SHIP_INFO": "船舶档案", "TALK": "闲聊",
                }
                print("\n【各意图准确率】")
                per_intent_acc = results.get("per_intent_accuracy", {})
                for intent, stats in sorted(per_intent_acc.items()):
                    cn_name = INTENT_LABELS_CN.get(intent, intent)
                    acc = stats["accuracy"]
                    correct = stats["correct"]
                    total = stats["total"]
                    bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                    print(f"  {cn_name:10s} [{bar}] {acc:.2%} ({correct}/{total})")

                bad_cases = results.get("bad_cases", [])
                if bad_cases:
                    print(f"\n【Bad Cases】(共 {len(bad_cases)} 条)")
                    for i, case in enumerate(bad_cases[:10], 1):
                        print(f"\n  #{i} ID: {case.get('id', 'N/A')}")
                        print(f"     Query: {case.get('query', '')[:50]}")
                        gold = case.get("gold", {})
                        pred = case.get("pred", {})
                        print(f"     预期: {INTENT_LABELS_CN.get(gold.get('intent',''), gold.get('intent',''))} | 预测: {INTENT_LABELS_CN.get(pred.get('intent',''), pred.get('intent',''))}")

                print("\n" + "=" * 60)
                print("评测完成")
                print("=" * 60 + "\n")

                # 保存评测报告到 checkpoint 目录
                report_path = checkpoint_dir / "evaluation_results.json"
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info("评测报告已保存到: %s", report_path)

            except Exception as e:
                logger.error("自动评测失败: %s", e)
        else:
            logger.warning("测试集不存在，跳过自动评测: %s", test_path)

    # 保存版本记录
    versions_file = outputs_base / VERSIONS_FILE
    if versions_file.exists():
        with versions_file.open("r", encoding="utf-8") as f:
            versions = json.load(f)
    else:
        versions = []

    versions.append(version)
    with versions_file.open("w", encoding="utf-8") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)
    logger.info("版本记录已更新: %s", versions_file)


if __name__ == "__main__":
    main()
