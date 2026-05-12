from __future__ import annotations

"""多模态输入服务。"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.config import get_settings
from backend.infra.database.repositories.oss_repository import OssRecord, OssRepository
from backend.infra.llm.client import LLMClient
from backend.services.oss_service import OssService
from backend.tools.order import OrderTool

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "aac", "amr", "ogg", "opus", "flac", "wma", "mp4", "pcm", "webm", "aiff"}


class InputServiceError(Exception):
    """输入处理错误。"""


class OCRService:
    """调用通义 OCR。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def image_to_text(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        data_url = self._build_data_url(image_bytes, mime_type)
        return await self._call_dashscope([{"image": data_url}])

    async def ocr_by_url(self, url: str, prompt: Optional[str] = None) -> str:
        content = [{"image": url}]
        if prompt:
            content.append({"text": prompt})
        return await self._call_dashscope(content)

    async def _call_dashscope(self, content: list[dict[str, str]]) -> str:
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.settings.tongyi_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "qwen-vl-ocr-latest",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            items = data["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InputServiceError(f"OCR 返回格式异常: {data}") from exc

        texts = [str(item.get("text", "")).strip() for item in items if isinstance(item, dict)]
        return "\n".join(part for part in texts if part).strip()

    def _build_data_url(self, image_bytes: bytes, mime_type: str) -> str:
        import base64

        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


class ASRService:
    """调用 Fabrx 识别音频。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "audio.bin",
        content_type: str = "application/octet-stream",
    ) -> str:
        headers = {"X-API-Key": self.settings.fabrx_api_key}
        files = {
            "file": (filename, audio_bytes, content_type),
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(self.settings.fabrx_endpoint, headers=headers, files=files)
            response.raise_for_status()

        text = self._extract_text(response)
        return text.strip()

    def _extract_text(self, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = response.json()
            for key in ("text", "content", "result", "recognizedText", "recognized_text"):
                value = body.get(key) if isinstance(body, dict) else None
                if isinstance(value, str) and value.strip():
                    return value
            if isinstance(body, dict):
                nested = body.get("data")
                if isinstance(nested, dict):
                    for key in ("text", "content", "result", "recognizedText", "recognized_text"):
                        value = nested.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
            return json.dumps(body, ensure_ascii=False)

        return response.text


class InputService:
    """对齐 Java InputService 的 Python 版本。"""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        ocr_service: OCRService | None = None,
        asr_service: ASRService | None = None,
        oss_repository: OssRepository | None = None,
        oss_service: OssService | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.ocr_service = ocr_service or OCRService()
        self.asr_service = asr_service or ASRService()
        self.oss_repository = oss_repository or OssRepository()
        self.oss_service = oss_service or OssService()

    async def recognize_image(self, content: bytes, filename: str, content_type: str) -> str:
        text = await self.ocr_service.image_to_text(content, mime_type=content_type or self._guess_mime_type(filename))
        if not text:
            raise InputServiceError("图片识别结果为空")
        return text

    async def recognize_speech(self, content: bytes, filename: str, content_type: str) -> str:
        text = await self.asr_service.speech_to_text(content, filename=filename, content_type=content_type)
        if not text:
            raise InputServiceError("语音识别结果为空")
        return text

    async def recognize_by_oss_id(self, oss_id: int) -> str:
        record = await self.oss_repository.find_by_id(oss_id)
        if record is None:
            raise InputServiceError("未找到对应OSS文件")

        extension = self._resolve_extension(record)
        if extension in IMAGE_EXTENSIONS:
            return await self.ocr_service.ocr_by_url(
                record.url if record.url.startswith("http") else self._local_url(record),
                "Please output only the text content from the image.",
            )

        if extension in AUDIO_EXTENSIONS:
            path = self.oss_service.resolve_file_path(record)
            content = path.read_bytes()
            return await self.recognize_speech(content, record.original_name or path.name, self._guess_mime_type(path.name))

        raise InputServiceError(f"暂不支持的文件类型: {extension}")

    async def analysis_order_params(self, raw_text: str) -> dict[str, Any]:
        preview = await OrderTool().run({"text": raw_text, "action": "preview"})
        goods_count_raw = preview.get("tons")
        try:
            goods_count = int(goods_count_raw) if goods_count_raw is not None else None
        except (TypeError, ValueError):
            goods_count = None

        load_name = preview.get("loading_port")
        unload_name = preview.get("discharge_port")
        return {
            "success": True,
            "message": "解析成功",
            "recognizedText": raw_text,
            "data": {
                "loadAddressName": load_name,
                "loadAddressLon": None,
                "loadAddressLat": None,
                "loadCityName": self._guess_city_name(load_name),
                "unloadAddressName": unload_name,
                "unloadAddressLon": None,
                "unloadAddressLat": None,
                "unloadCityName": self._guess_city_name(unload_name),
                "loadDate": preview.get("shipping_date"),
                "goodsInfoName": preview.get("cargo_name"),
                "goodsCount": goods_count,
                "goodsUnit": 1,
                "goodsUnitName": "吨",
            },
        }

    async def shipping_route(self, raw_text: str) -> dict[str, Any]:
        parsed = await self._extract_shipping_route(raw_text)
        return {
            "loadAddressName": parsed.get("loadAddressName"),
            "loadAddressLon": None,
            "loadAddressLat": None,
            "loadCityName": self._guess_city_name(parsed.get("loadAddressName")),
            "unloadAddressName": parsed.get("unloadAddressName"),
            "unloadAddressLon": None,
            "unloadAddressLat": None,
            "unloadCityName": self._guess_city_name(parsed.get("unloadAddressName")),
        }

    async def shipping_route_by_audio(self, content: bytes, filename: str, content_type: str) -> dict[str, Any]:
        raw_text = await self.recognize_speech(content, filename, content_type)
        return await self.shipping_route(raw_text)

    async def _extract_shipping_route(self, raw_text: str) -> dict[str, str | None]:
        prompt = (
            "你是航运航线信息提取助手。"
            "请从用户文本中提取装货地址和卸货地址，并严格输出 JSON，"
            "格式为 {\"loadAddressName\":\"...\",\"unloadAddressName\":\"...\"}。"
            "如果缺失则填 null，不要输出额外说明。"
        )
        try:
            content = await self.llm_client.chat(system_prompt=prompt, user_message=raw_text)
            data = self._safe_parse_json(content)
            if isinstance(data, dict):
                load_name = self._clean_optional_str(data.get("loadAddressName"))
                unload_name = self._clean_optional_str(data.get("unloadAddressName"))
                if load_name or unload_name:
                    return {
                        "loadAddressName": load_name,
                        "unloadAddressName": unload_name,
                    }
        except Exception as exc:
            logger.warning("shipping route llm parse failed: %s", exc)

        return self._extract_shipping_route_by_regex(raw_text)

    def _extract_shipping_route_by_regex(self, raw_text: str) -> dict[str, str | None]:
        patterns = [
            r"(?P<load>[\u4e00-\u9fa5A-Za-z0-9]{2,20})到(?P<unload>[\u4e00-\u9fa5A-Za-z0-9]{2,20})",
            r"从(?P<load>[\u4e00-\u9fa5A-Za-z0-9]{2,20})(?:装|出发).*到(?P<unload>[\u4e00-\u9fa5A-Za-z0-9]{2,20})",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_text)
            if match:
                return {
                    "loadAddressName": self._clean_optional_str(match.group("load")),
                    "unloadAddressName": self._clean_optional_str(match.group("unload")),
                }
        return {
            "loadAddressName": None,
            "unloadAddressName": None,
        }

    def _local_url(self, record: OssRecord) -> str:
        return f"http://127.0.0.1:9020{record.url}"

    def _resolve_extension(self, record: OssRecord) -> str:
        candidates = [
            record.file_suffix,
            Path(record.original_name).suffix,
            Path(record.file_name).suffix,
            Path(record.url).suffix,
        ]
        for candidate in candidates:
            normalized = self._normalize_extension(candidate)
            if normalized:
                return normalized
        raise InputServiceError("无法识别文件类型，请检查文件后缀")

    def _normalize_extension(self, extension: Optional[str]) -> str:
        value = (extension or "").strip().lower()
        if value.startswith("."):
            value = value[1:]
        return value

    def _guess_mime_type(self, filename: str) -> str:
        import mimetypes

        media_type, _ = mimetypes.guess_type(filename)
        return media_type or "application/octet-stream"

    def _safe_parse_json(self, content: str) -> Any:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text).strip()
        return json.loads(text)

    def _clean_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _guess_city_name(self, address: Any) -> str | None:
        text = self._clean_optional_str(address)
        if not text:
            return None
        for token in ("市", "港", "区", "县"):
            index = text.find(token)
            if index > 0:
                return text[: index + 1]
        return text
