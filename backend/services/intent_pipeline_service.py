from __future__ import annotations

"""意图判定流水线服务。"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.graph.state.agent_state import (
    ClarifyInfo,
    EntityCandidate,
    IntentCandidate,
    SceneEnum,
    SlotInfo,
)
from train.slot_extractor import extract_slots


@dataclass
class PipelineResult:
    """统一返回给工作流和测试接口的判定结果。"""

    preprocessed_input: str
    intent_candidates: list[IntentCandidate]
    slots: dict[str, SlotInfo]
    entity_candidates: list[EntityCandidate]
    clarify: ClarifyInfo
    final_intent: str | None
    final_slots: dict[str, list[str]]
    can_route: bool


class IntentModelRuntime:
    """懒加载训练好的联合模型，并返回运行时推理结果。"""

    def __init__(self) -> None:
        self._loaded = False
        self._available = False
        self._tokenizer = None
        self._model = None
        self._device = None

    def _checkpoint_dir(self) -> Path | None:
        base_dir = Path(__file__).resolve().parents[2] / "train" / "outputs"
        if not base_dir.exists():
            return None
        versions = sorted(
            [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("V")],
            key=lambda item: item.name,
        )
        for version_dir in reversed(versions):
            checkpoint_dir = version_dir / "checkpoints"
            if (checkpoint_dir / "best_model.bin").exists():
                return checkpoint_dir
        return None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch
            from transformers import AutoTokenizer

            from train.common import BIO_LABELS, INTENT_LABELS, MODEL_NAME, build_model_text, decode_bio_entities
            from train.scripts.train import BertIntentClassifier
        except Exception:
            self._available = False
            return

        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None:
            self._available = False
            return

        metadata_path = checkpoint_dir / "metadata.json"
        model_name = MODEL_NAME
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_name = str(metadata.get("model_name", MODEL_NAME))

        tokenizer_dir = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
        model = BertIntentClassifier(
            model_name=model_name,
            num_intents=len(INTENT_LABELS),
            num_bio_labels=len(BIO_LABELS),
        )
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.bin", map_location=device))
        model.to(device)
        model.eval()

        self._torch = torch
        self._build_model_text = build_model_text
        self._decode_bio_entities = decode_bio_entities
        self._intent_labels = INTENT_LABELS
        self._checkpoint_dir = checkpoint_dir
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._available = True

    def predict(self, query: str, history: list[dict[str, str]]) -> dict[str, Any] | None:
        self._load()
        if not self._available or self._tokenizer is None or self._model is None or self._device is None:
            return None

        sample = {"history": history, "query": query}
        text = self._build_model_text(sample)
        encoding = self._tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)
        with self._torch.no_grad():
            intent_logits, slot_logits, clarify_logits = self._model(input_ids, attention_mask)

        intent_probs = self._torch.softmax(intent_logits[0], dim=-1).cpu().tolist()
        clarify_probs = self._torch.softmax(clarify_logits[0], dim=-1).cpu().tolist()
        top_indices = sorted(range(len(intent_probs)), key=lambda idx: intent_probs[idx], reverse=True)[:5]
        candidates = [
            {"intent": self._intent_labels[idx], "score": float(intent_probs[idx]), "source": "model"}
            for idx in top_indices
        ]

        bio_ids = slot_logits[0].argmax(dim=-1).cpu().tolist()
        offsets = [tuple(pair) for pair in encoding["offset_mapping"][0].cpu().tolist()]
        entities = self._decode_bio_entities(bio_ids, offsets, text)

        return {
            "text": text,
            "intent_candidates": candidates,
            "slot_entities": entities,
            "clarify_score": float(clarify_probs[1]),
        }


class IntentPipelineService:
    """按“规则预处理 -> TopK -> 槽位/实体 -> 一致性 -> 澄清”的流程产出结果。"""

    RULE_INTENTS = (
        ("天气", SceneEnum.QUERY_WEATHER),
        ("能见度", SceneEnum.QUERY_WEATHER),
        ("水位", SceneEnum.QUERY_WATER_LEVEL),
        ("查船", SceneEnum.QUERY_SHIP),
        ("轨迹", SceneEnum.QUERY_SHIP),
        ("位置", SceneEnum.QUERY_SHIP),
        ("到哪", SceneEnum.QUERY_SHIP),
        ("找船", SceneEnum.FIND_SHIP),
        ("运价", SceneEnum.QUERY_FREIGHT),
        ("运费", SceneEnum.QUERY_FREIGHT),
        ("运单", SceneEnum.SAVE_ORDER),
        ("录单", SceneEnum.SAVE_ORDER),
        ("订单", SceneEnum.QUERY_ORDER),
        ("投诉", SceneEnum.FEEDBACK),
        ("客服态度", SceneEnum.FEEDBACK),
        ("识别", SceneEnum.IMAGE_OCR),
    )

    REQUIRED_SLOTS = {
        SceneEnum.QUERY_WEATHER: (("route_from", "port_name"),),
        SceneEnum.QUERY_WATER_LEVEL: (("route_from", "port_name", "area_name"),),
        SceneEnum.FIND_SHIP: (("route_from", "port_name"), ("route_to", "port_name")),
        SceneEnum.SAVE_ORDER: (("route_from", "port_name"), ("route_to", "port_name"), ("cargo_name",), ("cargo_weight",)),
    }

    CLARIFY_TEMPLATES = {
        SceneEnum.QUERY_WEATHER: "你想查哪条航线或哪个港口的天气？",
        SceneEnum.QUERY_WATER_LEVEL: "你想看哪条航线、哪个港口，或者哪个区域的水位？",
        SceneEnum.FIND_SHIP: "请补充起点和终点，我再帮你找合适的船。",
        SceneEnum.SAVE_ORDER: "请补充装货地、卸货地、货物名称和吨位，我再帮你生成运单。",
        SceneEnum.QUERY_SHIP: "请告诉我船名，我再帮你查位置或轨迹。",
        SceneEnum.QUERY_ORDER: "请告诉我想查哪一票运单，或者给我订单号。",
        SceneEnum.TALK: "你是想查什么业务？比如查船、天气、运价或录入运单。",
    }

    def __init__(self) -> None:
        self._runtime = IntentModelRuntime()

    async def analyze(self, user_input: str, history: list[dict[str, str]] | None = None) -> PipelineResult:
        history = history or []
        normalized_text = self._preprocess(user_input)
        rule_candidates = self._rule_candidates(normalized_text)
        model_result = self._runtime.predict(normalized_text, history)

        intent_candidates = self._merge_candidates(rule_candidates, model_result)
        slot_map = self._merge_slots(normalized_text, model_result)
        entity_candidates = self._recall_entities(slot_map)
        clarify = self._decide_clarify(intent_candidates, slot_map, model_result)
        final_intent = None if clarify.need_clarify else (intent_candidates[0].intent if intent_candidates else SceneEnum.TALK)
        final_slots = {name: info.values for name, info in slot_map.items() if info.values}

        return PipelineResult(
            preprocessed_input=normalized_text,
            intent_candidates=intent_candidates,
            slots=slot_map,
            entity_candidates=entity_candidates,
            clarify=clarify,
            final_intent=final_intent,
            final_slots=final_slots,
            can_route=final_intent is not None,
        )

    def _preprocess(self, text: str) -> str:
        return " ".join(text.replace("，", " ").replace("。", " ").split())

    def _rule_candidates(self, text: str) -> list[IntentCandidate]:
        hits: dict[str, float] = defaultdict(float)
        for keyword, intent in self.RULE_INTENTS:
            if keyword in text:
                hits[intent] += 0.34

        candidates = [
            IntentCandidate(intent=intent, score=min(score, 0.98), source="rule")
            for intent, score in hits.items()
        ]
        if not candidates:
            candidates.append(IntentCandidate(intent=SceneEnum.TALK, score=0.4, source="fallback"))
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:5]

    def _merge_candidates(
        self,
        rule_candidates: list[IntentCandidate],
        model_result: dict[str, Any] | None,
    ) -> list[IntentCandidate]:
        merged: dict[str, IntentCandidate] = {}
        for candidate in rule_candidates:
            merged[candidate.intent] = candidate

        if model_result:
            for candidate in model_result["intent_candidates"]:
                runtime_intent = SceneEnum.from_model_intent(candidate["intent"])
                if runtime_intent is None:
                    continue
                score = float(candidate["score"])
                previous = merged.get(runtime_intent)
                if previous is None or score > previous.score:
                    merged[runtime_intent] = IntentCandidate(
                        intent=runtime_intent,
                        score=score,
                        source=candidate["source"],
                    )

        if SceneEnum.TALK not in merged:
            merged[SceneEnum.TALK] = IntentCandidate(intent=SceneEnum.TALK, score=0.2, source="fallback")
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:5]

    def _merge_slots(self, text: str, model_result: dict[str, Any] | None) -> dict[str, SlotInfo]:
        slot_values: dict[str, list[str]] = {}
        slot_confidence: dict[str, float] = {}
        slot_source: dict[str, str] = {}

        rule_slots = extract_slots(text)
        for slot_name, values in rule_slots.items():
            cleaned_values = [value.strip() for value in values if str(value).strip()]
            if cleaned_values:
                slot_values[slot_name] = cleaned_values
                slot_confidence[slot_name] = 0.82
                slot_source[slot_name] = "rule"

        if model_result:
            for entity in model_result["slot_entities"]:
                slot_name = str(entity["slot"])
                value = str(entity["text"]).strip()
                if not value:
                    continue
                slot_values.setdefault(slot_name, [])
                if value not in slot_values[slot_name]:
                    slot_values[slot_name].append(value)
                slot_confidence[slot_name] = max(slot_confidence.get(slot_name, 0.0), 0.75)
                slot_source[slot_name] = "hybrid" if slot_name in rule_slots and rule_slots[slot_name] else "model"

        return {
            slot_name: SlotInfo(
                name=slot_name,
                values=values,
                source=slot_source.get(slot_name, "rule"),
                confidence=slot_confidence.get(slot_name, 0.0),
            )
            for slot_name, values in sorted(slot_values.items())
        }

    def _recall_entities(self, slots: dict[str, SlotInfo]) -> list[EntityCandidate]:
        entities: list[EntityCandidate] = []
        for slot_name, slot_info in slots.items():
            entity_type = self._entity_type(slot_name)
            for value in slot_info.values:
                entities.append(
                    EntityCandidate(
                        slot_name=slot_name,
                        value=value,
                        entity_type=entity_type,
                        score=slot_info.confidence,
                    )
                )
        return entities

    def _entity_type(self, slot_name: str) -> str:
        if slot_name == "ship_name":
            return "ship"
        if slot_name in {"route_from", "route_to", "port_name", "area_name"}:
            return "location"
        if slot_name in {"cargo_name", "cargo_weight"}:
            return "cargo"
        if slot_name == "date_time":
            return "time"
        return "generic"

    def _decide_clarify(
        self,
        candidates: list[IntentCandidate],
        slots: dict[str, SlotInfo],
        model_result: dict[str, Any] | None,
    ) -> ClarifyInfo:
        if not candidates:
            return ClarifyInfo(
                need_clarify=True,
                question=self.CLARIFY_TEMPLATES[SceneEnum.TALK],
                reason="no_intent",
            )

        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None

        if top2 and abs(top1.score - top2.score) < 0.12:
            return ClarifyInfo(
                need_clarify=True,
                question="我识别到不止一个可能意图，你是想查船、查天气，还是录入运单？",
                reason="intent_ambiguous",
            )

        if model_result and float(model_result.get("clarify_score", 0.0)) >= 0.5:
            return ClarifyInfo(
                need_clarify=True,
                question=self.CLARIFY_TEMPLATES.get(top1.intent, self.CLARIFY_TEMPLATES[SceneEnum.TALK]),
                reason="model_clarify",
            )

        missing_groups = self._missing_required_slots(top1.intent, slots)
        if missing_groups:
            return ClarifyInfo(
                need_clarify=True,
                question=self.CLARIFY_TEMPLATES.get(top1.intent, self.CLARIFY_TEMPLATES[SceneEnum.TALK]),
                reason="missing_slots",
            )

        return ClarifyInfo(need_clarify=False)

    def _missing_required_slots(self, intent: str, slots: dict[str, SlotInfo]) -> list[tuple[str, ...]]:
        requirements = self.REQUIRED_SLOTS.get(intent, ())
        missing: list[tuple[str, ...]] = []
        for group in requirements:
            if not any(slots.get(slot_name) and slots[slot_name].values for slot_name in group):
                missing.append(group)
        return missing
