from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse

from .config import get_settings
from .schemas import VisionResult

settings = get_settings()


class QwenVisionRuntime:
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.lock = asyncio.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError("install the 'vision' optional dependencies") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the local vision service")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            settings.vision_model_name,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
        )
        if settings.vision_adapter_path:
            self.model = PeftModel.from_pretrained(
                base_model,
                settings.vision_adapter_path,
                is_trainable=False,
            )
        else:
            self.model = base_model
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            settings.vision_model_name,
            trust_remote_code=True,
        )

    async def recognize(self, content: bytes) -> VisionResult:
        async with self.lock:
            return await asyncio.to_thread(self._recognize_sync, content)

    def _recognize_sync(self, content: bytes) -> VisionResult:
        self.load()
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(BytesIO(content)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "识别图片中所有可以可靠确认的食材。只输出 JSON："
                            "{items:[{name,normalized_name,confidence}]}。"
                            "不要识别日期、数量、包装规格或非食材；无法确认的物体不要猜测。"
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=900, do_sample=False)
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated, strict=True)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("vision model did not return JSON")
        data = json.loads(match.group(0))
        data["model_version"] = settings.vision_model_version
        data["raw_text"] = raw
        return VisionResult.model_validate(data)


runtime = QwenVisionRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="self-evolution-Agent Vision",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> ORJSONResponse:
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    return ORJSONResponse(
        {
            "status": "ok" if cuda else "not_ready",
            "cuda": cuda,
            "model_loaded": runtime.model is not None,
        },
        status_code=200 if cuda else 503,
    )


@app.post("/v1/recognize", response_model=VisionResult)
async def recognize(image: UploadFile) -> VisionResult:
    content = await image.read(settings.vision_max_image_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="empty image")
    if len(content) > settings.vision_max_image_bytes:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        return await runtime.recognize(content)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
