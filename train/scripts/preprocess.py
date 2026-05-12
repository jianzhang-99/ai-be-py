"""
数据预处理脚本

功能：
1. 将原始对话数据转换为模型训练所需的 JSONL 格式
2. 支持从 ai_chat_log 表的历史数据导出
3. 数据集划分（train/valid/test）

用法：
    python -m train.scripts.preprocess --input ./data/raw/chat_history.jsonl --output ./data/processed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from train.common import log_validation_result, read_jsonl, validate_samples, write_jsonl

# Intent 标签映射（Java 版本 -> 我们的版本）
INTENT_MAPPING = {
    "QUERY_SHIP": "QUERY_SHIP",
    "FIND_SHIP": "FIND_SHIP",
    "QUERY_FREIGHT": "QUERY_FREIGHT",
    "TALK": "TALK",
    "DOC_QA": "DOC_QA",
    "SAVE_ORDER": "SAVE_ORDER",
    "QUERY_ORDER": "QUERY_ORDER",
    "DISPATCH_MONITOR": "DISPATCH_MONITOR",
    "IMAGE_OCR": "IMAGE_OCR",
    "FEEDBACK": "FEEDBACK",
    "QUERY_WEATHER": "QUERY_WEATHER",
    "QUERY_WATER_LEVEL": "QUERY_WATER_LEVEL",
    "QUERY_OIL_STATION": "QUERY_OIL_STATION",
    "QUERY_SHIP_INFO": "QUERY_SHIP_INFO",
    # 以下是 Java 版本可能有的标签，映射到最接近的类别
    "CHAT": "TALK",
    "OTHER": "TALK",
    "UNKNOWN": "TALK",
}

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> list[dict]:
    """加载原始数据"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"跳过格式错误行: {line[:50]}")
    logger.info(f"加载原始数据 {len(samples)} 条")
    return samples


def convert_to_jsonl(samples: list[dict], output_path: str, intent_mapping: dict = None) -> int:
    """将样本转换为标准 JSONL 格式"""
    if intent_mapping is None:
        intent_mapping = INTENT_MAPPING

    intent_counts = defaultdict(int)
    converted = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # 解析原始数据
            user_input = sample.get("user_input", "")
            ai_response = sample.get("ai_response", "")
            intent_code = sample.get("intent_code", sample.get("intent", "TALK"))
            session_id = sample.get("session_id", "")
            history = sample.get("history", [])

            # 映射 intent
            intent = intent_mapping.get(intent_code, intent_code)
            if intent not in INTENT_MAPPING.values():
                intent = "TALK"

            # 构建标准格式
            converted_sample = {
                "id": f"sample_{converted+1:06d}",
                "history": history[-3:] if history else [],  # 最多 3 轮
                "query": user_input,
                "label": {
                    "intent": intent,
                    "slots": {},  # 历史数据可能没有槽位标注，需人工补充
                    "need_clarify": False,
                },
                "metadata": {
                    "session_id": session_id,
                    "ai_response": ai_response,
                }
            }

            f.write(json.dumps(converted_sample, ensure_ascii=False) + "\n")
            intent_counts[intent] += 1
            converted += 1

    logger.info(f"转换完成 {converted} 条，保存到 {output_path}")
    logger.info(f"意图分布: {dict(intent_counts)}")
    return converted


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 42,
):
    """划分训练集/验证集/测试集"""
    samples = read_jsonl(input_path)

    # 按 intent 分层抽样，确保每个 intent 在各数据集中都有样本
    intent_samples = defaultdict(list)
    for sample in samples:
        intent = sample["label"]["intent"]
        intent_samples[intent].append(sample)

    train_samples = []
    valid_samples = []
    test_samples = []

    for intent, intent_list in intent_samples.items():
        random.seed(seed)
        random.shuffle(intent_list)

        n = len(intent_list)
        if n >= 3:
            n_valid = max(1, int(n * valid_ratio))
            n_test = max(1, n - int(n * train_ratio) - n_valid)
            n_train = n - n_valid - n_test
        elif n == 2:
            logger.warning("%s 只有 2 条样本，只能分到 train/test，valid 会缺该类", intent)
            n_train = 1
            n_valid = 0
        else:
            logger.warning("%s 只有 1 条样本，只能分到 train，valid/test 会缺该类", intent)
            n_train = 1
            n_valid = 0

        train_samples.extend(intent_list[:n_train])
        valid_samples.extend(intent_list[n_train:n_train + n_valid])
        test_samples.extend(intent_list[n_train + n_valid:])

    # 打乱
    random.seed(seed)
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(test_samples)

    # 重新编号
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
        filepath = output_path / f"{name}.jsonl"
        for i, sample in enumerate(data):
            sample["id"] = f"{name}_{i+1:06d}"
        write_jsonl(data, filepath)
        logger.info(f"{name}.jsonl: {len(data)} 条")

    logger.info(f"划分完成: train={len(train_samples)}, valid={len(valid_samples)}, test={len(test_samples)}")


def merge_jsonl_files(input_dir: str, output_path: str, pattern: str = "*.jsonl") -> int:
    """合并目录下多个 JSONL 文件，常用于把按 intent 拆分的数据合成总训练集。"""
    input_path = Path(input_dir)
    files = sorted(input_path.glob(pattern))
    aggregate_names = {
        "all",
        "全部样本",
        "训练集",
        "训练集_真实样本",
        "train",
        "valid",
        "test",
        "ambiguity_test",
        "evaluation_results",
    }
    if len(files) > 1:
        # 目录里常同时存在“按 intent 拆分文件”和“已合并文件”，这里默认跳过已合并文件，
        # 避免 build 时把同一批样本重复合进去。
        files = [filepath for filepath in files if filepath.stem not in aggregate_names]
    if not files:
        raise FileNotFoundError(f"目录中没有匹配文件: {input_path}/{pattern}")

    samples: list[dict] = []
    seen_ids: set[str] = set()
    for filepath in files:
        for sample in read_jsonl(filepath):
            sample_id = str(sample.get("id", ""))
            if sample_id in seen_ids:
                # 合并时自动补上文件名前缀，避免不同文件里的 sample_0001 冲突。
                sample["id"] = f"{filepath.stem}_{sample_id}"
            seen_ids.add(str(sample.get("id", "")))
            samples.append(sample)

    write_jsonl(samples, output_path)
    logger.info("合并 %s 个文件，共 %s 条样本 -> %s", len(files), len(samples), output_path)
    return len(samples)


def build_dataset(input_dir: str, output_dir: str, pattern: str = "*.jsonl", seed: int = 42):
    """合并、校验并划分 train/valid/test，一步完成训练前数据准备。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_path = output_path / "all.jsonl"
    merge_jsonl_files(input_dir=input_dir, output_path=str(merged_path), pattern=pattern)

    samples = read_jsonl(merged_path)
    result = validate_samples(samples, source_name=str(merged_path))
    log_validation_result(result)
    if not result.ok:
        logger.error("数据存在错误，已停止划分。请先修复错误后再执行 build。")
        sys.exit(1)

    split_dataset(str(merged_path), str(output_path), seed=seed)


def validate_jsonl_file(filepath: str) -> bool:
    """校验 JSONL 文件格式，错误返回 False。"""
    result = validate_samples(read_jsonl(filepath), source_name=filepath)
    log_validation_result(result)
    return result.ok


def generate_sample_queries(output_path: str, intent: str, count: int = 50):
    """生成指定 intent 的示例样本（用于快速测试）"""
    import random

    # 基础 query 模板
    templates = {
        "QUERY_SHIP": [
            "{ship_name}在哪",
            "查一下{ship_name}的位置",
            "{ship_name}现在到哪了",
            "{ship_name}预计什么时候到港",
            "这条船的轨迹",
        ],
        "FIND_SHIP": [
            "{area_name}附近有没有船",
            "帮我找一条从{from_place}到{to_place}的船",
            "{area_name}{cargo_weight}的船有吗",
            "长江下游有空的船吗",
            "{port_name}能跑{cargo_weight}的船吗",
        ],
        "QUERY_FREIGHT": [
            "{from_place}到{to_place}{cargo_name}{cargo_weight}运价多少",
            "{from_place}到{to_place}砂石运价",
            "{cargo_weight}船从{from_place}到{to_place}多少钱",
            "现在煤炭运费行情",
        ],
        "TALK": [
            "你好",
            "在吗",
            "早上好",
            "帮我查一下",
            "你们这个怎么用",
        ],
    }

    ship_names = ["俞垛79", "华航118", "长江之星6号", "南京7", "武穴号"]
    area_names = ["南京港附近", "长江下游", "江北一带", "南通附近"]
    port_names = ["南京龙潭港", "南通港", "镇江港", "重庆果园港"]
    from_places = ["南京", "武汉", "南通", "靖江"]
    to_places = ["重庆", "上海", "南京", "南通"]
    cargo_names = ["砂石", "煤炭", "钢材", "集装箱"]
    cargo_weights = ["3000吨", "5000吨", "8000吨", "1万吨"]

    if intent not in templates:
        logger.warning(f"未知 intent: {intent}")
        return

    samples = []
    for i in range(count):
        template = random.choice(templates[intent])
        query = template.format(
            ship_name=random.choice(ship_names),
            area_name=random.choice(area_names),
            port_name=random.choice(port_names),
            from_place=random.choice(from_places),
            to_place=random.choice(to_places),
            cargo_name=random.choice(cargo_names),
            cargo_weight=random.choice(cargo_weights),
        )

        sample = {
            "id": f"sample_{intent.lower()}_{i+1:04d}",
            "history": [],
            "query": query,
            "label": {
                "intent": intent,
                "slots": {},
                "need_clarify": False,
            },
        }
        samples.append(sample)

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"生成 {count} 条 {intent} 样本，保存到 {output_path}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["convert", "split", "generate", "merge", "build", "validate"],
                        help="模式: convert=转换格式, split=划分数据集, generate=生成示例, merge=合并, build=合并校验划分, validate=校验")
    parser.add_argument("--input", type=str, help="输入文件")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--input_dir", type=str, help="输入目录，merge/build 模式使用")
    parser.add_argument("--pattern", type=str, default="*.jsonl", help="输入文件匹配规则")
    parser.add_argument("--intent", type=str, help="generate 模式时指定 intent 类型")
    parser.add_argument("--count", type=int, default=50, help="generate 模式时生成数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.mode == "convert":
        if not args.input or not args.output:
            logger.error("convert 模式需要 --input 和 --output")
            sys.exit(1)
        samples = load_raw_data(args.input)
        convert_to_jsonl(samples, args.output)

    elif args.mode == "split":
        if not args.input or not args.output:
            logger.error("split 模式需要 --input 和 --output")
            sys.exit(1)
        split_dataset(args.input, args.output, seed=args.seed)

    elif args.mode == "generate":
        if not args.output or not args.intent:
            logger.error("generate 模式需要 --output 和 --intent")
            sys.exit(1)
        generate_sample_queries(args.output, args.intent, args.count)

    elif args.mode == "merge":
        if not args.input_dir or not args.output:
            logger.error("merge 模式需要 --input_dir 和 --output")
            sys.exit(1)
        merge_jsonl_files(args.input_dir, args.output, args.pattern)

    elif args.mode == "build":
        if not args.input_dir or not args.output:
            logger.error("build 模式需要 --input_dir 和 --output")
            sys.exit(1)
        build_dataset(args.input_dir, args.output, args.pattern, args.seed)

    elif args.mode == "validate":
        if not args.input:
            logger.error("validate 模式需要 --input")
            sys.exit(1)
        ok = validate_jsonl_file(args.input)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
