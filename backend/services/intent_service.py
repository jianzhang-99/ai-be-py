from __future__ import annotations

"""最小改动版意图识别：规则预处理 + 专属模型。"""

from pathlib import Path
from typing import Any, Optional

from backend.graph.state.agent_state import SceneEnum


class CustomIntentModelRuntime:
    """本地加载训练好的专属意图识别模型。"""

    def __init__(self) -> None:
        self._loaded = False
        self._available = False
        self._tokenizer = None
        self._model = None
        self._device = None

    def _checkpoint_dir(self) -> Path | None:
        outputs_dir = Path(__file__).resolve().parents[2] / "train" / "outputs"
        if not outputs_dir.exists():
            return None

        version_dirs = sorted(
            [path for path in outputs_dir.iterdir() if path.is_dir() and path.name.startswith("V")],
            key=lambda path: path.name,
        )
        for version_dir in reversed(version_dirs):
            checkpoint_dir = version_dir / "checkpoints"
            if (checkpoint_dir / "best_model.bin").exists():
                return checkpoint_dir
        return None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        try:
            import json
            import torch
            from transformers import AutoTokenizer

            from train.common import BIO_LABELS, INTENT_LABELS, MODEL_NAME, build_model_text
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
        self._intent_labels = INTENT_LABELS
        self._build_model_text = build_model_text
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._available = True

    def predict(self, user_input: str, history: list[dict[str, str]]) -> Optional[dict[str, Any]]:
        self._load()
        if not self._available or self._tokenizer is None or self._model is None or self._device is None:
            return None

        sample = {"history": history, "query": user_input}
        text = self._build_model_text(sample)
        encoding = self._tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with self._torch.no_grad():
            intent_logits, _, _ = self._model(input_ids, attention_mask)

        intent_probs = self._torch.softmax(intent_logits[0], dim=-1).cpu().tolist()
        top_index = max(range(len(intent_probs)), key=lambda idx: intent_probs[idx])
        return {
            "intent": SceneEnum.from_model_intent(self._intent_labels[top_index]),
            "confidence": float(intent_probs[top_index]),
        }


class IntentService:
    """意图识别服务。"""

    KEYWORD_RULES = (
        ("天气", SceneEnum.QUERY_WEATHER),
        ("查船", SceneEnum.QUERY_SHIP),
        ("船舶", SceneEnum.QUERY_SHIP),
        ("运单", SceneEnum.SAVE_ORDER),
        ("录单", SceneEnum.SAVE_ORDER),
        ("找船", SceneEnum.FIND_SHIP),
        ("水位", SceneEnum.QUERY_WATER_LEVEL),
        ("运价", SceneEnum.QUERY_FREIGHT),
    )

    def __init__(self) -> None:
        self.model_runtime = CustomIntentModelRuntime()

    async def recognize_by_rule(
        self,
        user_input: str,
        working_memory: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """规则命中时直接返回场景。"""

        for keyword, intent in self.KEYWORD_RULES:
            if keyword in user_input:
                return {"intent": intent, "confidence": 1.0, "method": "rule"}

        previous_scene = working_memory.get("current_scene")
        if previous_scene in {SceneEnum.QUERY_SHIP, SceneEnum.SAVE_ORDER}:
            if working_memory.get("state") == SceneEnum.WAITING_USER:
                return {"intent": previous_scene, "confidence": 1.0, "method": "rule"}

        return None

    async def recognize_by_model(
        self,
        user_input: str,
        history: list[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        """规则未命中时，使用专属训练模型输出意图。"""

        result = self.model_runtime.predict(user_input, history)
        if result is None or result.get("intent") is None:
            return None
        return {
            "intent": result["intent"],
            "confidence": result.get("confidence", 0.0),
            "method": "custom_model",
        }
