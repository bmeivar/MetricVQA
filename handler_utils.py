#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handler_utils.py

VLM handler infrastructure for metric_vqa.
Contains all model handler classes and the load_vlm factory function.
Imported by vlm_eval.py, reasoning_eval.py, and other eval scripts.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Type

from PIL import Image

# HF imports
try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
    )
    # Qwen2_5_VLForConditionalGeneration only exists in transformers >= 4.49.
    # Older envs (vlm-internvl2: 4.44, vlm-molmo: 4.49.0-rc) may not have it.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
    # AutoModelForImageTextToText only exists in transformers >= 4.46.
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        Blip2Processor = Blip2ForConditionalGeneration = None
    try:
        from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
    except ImportError:
        LlavaForConditionalGeneration = LlavaNextForConditionalGeneration = None
    try:
        from transformers import LlavaOnevisionForConditionalGeneration
    except ImportError:
        LlavaOnevisionForConditionalGeneration = None
    try:
        from transformers import PaliGemmaForConditionalGeneration
    except ImportError:
        PaliGemmaForConditionalGeneration = None
except Exception as e:
    raise ImportError(
        "Missing required ML dependencies. Install: torch, torchvision, transformers, huggingface_hub."
    ) from e


# -----------------------------
# Torch dtype helper
# -----------------------------

def _torch_dtype_from_str(s: str):
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    return None  # auto


def _get_device_map(device: int) -> str:
    if device < 0:
        return "cpu"
    return "auto"


def _get_device(device: int) -> torch.device:
    if device < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{device}")


# -----------------------------
# VLM Handler Base Class
# -----------------------------

class BaseVLMHandler(ABC):
    """Abstract base class for VLM model handlers."""

    def __init__(
        self,
        model_name: str,
        device: int,
        dtype: str,
        trust_remote_code: bool = False,
        use_flash_attn: bool = False,
        use_lora: bool = False,
        lora_path: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.use_flash_attn = use_flash_attn
        self.model = None
        self.processor = None
        self.torch_dtype = _torch_dtype_from_str(dtype)

    @abstractmethod
    def load(self) -> None:
        """Load model and processor."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate text response for image + prompt."""
        pass

    def format_prompt(self, instruction: str, question: str) -> str:
        """Format the full prompt from instruction and question."""
        return f"{instruction}\n{question}"

    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate without an image — pure LM behavior, no vision tokens.

        Default implementation uses the processor's chat template with a
        text-only message. Handlers whose models require image tokens in the
        prompt (PaliGemma, Phi-3.5) or have custom inference APIs (InternVL2,
        Molmo) override this method.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Handler not loaded.")

        # Build a text-only chat-template prompt.
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            # Fallback: raw prompt
            text = prompt

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        inputs = tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        if hasattr(self.processor, "decode"):
            decoded = self.processor.decode(generated, skip_special_tokens=True)
        else:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return decoded.strip()

    def _get_gen_kwargs(
        self,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> Dict[str, Any]:
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False
        return gen_kwargs

    def _get_attn_impl(self) -> Optional[str]:
        if self.use_flash_attn:
            return "flash_attention_2"
        return None


# -----------------------------
# VLM Handler Registry
# -----------------------------

VLM_REGISTRY: Dict[str, Type[BaseVLMHandler]] = {}


def register_handler(*patterns: str):
    """Decorator to register a handler for given model name patterns."""
    def decorator(cls: Type[BaseVLMHandler]) -> Type[BaseVLMHandler]:
        for pattern in patterns:
            VLM_REGISTRY[pattern.lower()] = cls
        return cls
    return decorator


def get_handler_class(model_name: str) -> Type[BaseVLMHandler]:
    """Get the appropriate handler class for a model name."""
    model_lower = model_name.lower()
    for pattern, handler_cls in VLM_REGISTRY.items():
        if pattern in model_lower:
            return handler_cls
    return GenericHandler


# -----------------------------
# Handler Implementations
# -----------------------------

@register_handler("llava-hf", "llava-v", "llava-1", "llava-next", "llava-onevision")
class LLaVAHandler(BaseVLMHandler):
    """Handler for LLaVA family models (llava-hf/*)."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.float16

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)
        kwargs["low_cpu_mem_usage"] = True

        model_lower = self.model_name.lower()
        if "onevision" in model_lower:
            from transformers import LlavaOnevisionForConditionalGeneration
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )
        elif "next" in model_lower or "v1.6" in model_lower or "1.6" in model_lower:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )

        if self.use_lora:
            print(f"Attaching LoRA weights from {self.lora_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        if image is None:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_prompt, return_tensors="pt"
            ).to(self.model.device)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_prompt, images=[image], return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("qwen2-vl", "qwen2.5-vl", "qwen/qwen2")
class QwenVLHandler(BaseVLMHandler):
    """Handler for Qwen-VL family models."""

    def load(self) -> None:
        from qwen_vl_utils import process_vision_info

        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = "auto"

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)

        print(f"Loading base Qwen model from {self.model_name}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )

        if self.use_lora:
            print(f"Attaching LoRA weights from {self.lora_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        if image is None:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_prompt], padding=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


@register_handler("blip2", "blip-2", "salesforce/blip2")
class BLIP2Handler(BaseVLMHandler):
    """Handler for BLIP-2 models."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.float16

        kwargs["device_map"] = _get_device_map(self.device)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = Blip2Processor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.model.device, self.torch_dtype or torch.float16)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()


@register_handler("smolvlm", "huggingfacetb/smolvlm")
class SmolVLMHandler(BaseVLMHandler):
    """Handler for SmolVLM models."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["_attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("idefics2", "idefics3", "huggingfacem4/idefics")
class Idefics2Handler(BaseVLMHandler):
    """Handler for Idefics2/3 models."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.float16

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["_attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("paligemma", "google/paligemma")
class PaliGemmaHandler(BaseVLMHandler):
    """Handler for PaliGemma models."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        kwargs["device_map"] = _get_device_map(self.device)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )

        if self.use_lora:
            print(f"Attaching LoRA weights from {self.lora_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        if image is None:
            inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("internvl2", "opengvlab/internvl")
class InternVL2Handler(BaseVLMHandler):
    """Handler for InternVL2 models (OpenGVLab/InternVL2-*)."""

    def load(self) -> None:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        kwargs["low_cpu_mem_usage"] = True
        kwargs["trust_remote_code"] = True
        kwargs["device_map"] = _get_device_map(self.device)

        if self.use_flash_attn:
            kwargs["use_flash_attn"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs
        )
        from transformers import GenerationMixin, GenerationConfig
        import inspect
        lm = getattr(self.model, 'language_model', None)
        if lm is not None and not isinstance(lm, GenerationMixin):
            orig_cls = lm.__class__
            lm.__class__ = type(
                orig_cls.__name__,
                (GenerationMixin, orig_cls),
                {}
            )
        if lm is not None and getattr(lm, 'generation_config', None) is None:
            lm.generation_config = GenerationConfig()
        if lm is not None:
            orig_forward = lm.__class__.forward
            sig = inspect.signature(orig_forward)
            accepted_params = set(sig.parameters.keys())
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_kwargs:
                import functools
                @functools.wraps(orig_forward)
                def _patched_forward(self_inner, *args, **kwargs):
                    filtered = {k: v for k, v in kwargs.items() if k in accepted_params}
                    return orig_forward(self_inner, *args, **filtered)
                lm.__class__.forward = _patched_forward
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )

        self._input_size = 448
        self._max_num = 6
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self._input_size, self._input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set()
        for n in range(min_num, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j <= max_num and i * j >= min_num:
                        target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = best_ratio[0] * image_size
        target_height = best_ratio[1] * image_size
        blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image(self, image):
        images = self._dynamic_preprocess(
            image, image_size=self._input_size, use_thumbnail=True, max_num=self._max_num
        )
        pixel_values = [self._transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        pixel_values = self._load_image(image).to(
            self.torch_dtype or torch.bfloat16
        ).to(self.model.device)

        question = f"<image>\n{prompt}"
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            generation_config["temperature"] = float(temperature)

        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response.strip()

    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """InternVL2 supports pure-text conversation via pixel_values=None.
        See https://huggingface.co/OpenGVLab/InternVL2-8B (pure-text example)."""
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            generation_config["temperature"] = float(temperature)
        response = self.model.chat(
            self.tokenizer, None, prompt, generation_config
        )
        return response.strip()


@register_handler("phi-3.5-vision", "phi-3-vision", "microsoft/phi-3")
class Phi35VisionHandler(BaseVLMHandler):
    """Handler for Phi-3.5-vision-instruct models."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.float16

        kwargs["trust_remote_code"] = True
        kwargs["device_map"] = _get_device_map(self.device)

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["_attn_implementation"] = attn_impl
        else:
            kwargs["_attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, num_crops=16
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{prompt}",
            }
        ]

        text_prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text_prompt, [image], return_tensors="pt").to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)
        gen_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


@register_handler("gemma-3", "google/gemma-3")
class Gemma3Handler(BaseVLMHandler):
    """Handler for Gemma 3 multimodal models."""

    def load(self) -> None:
        from transformers import Gemma3ForConditionalGeneration

        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        kwargs["device_map"] = _get_device_map(self.device)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        if image is None:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.torch_dtype or torch.bfloat16)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("molmo", "allenai/molmo")
class MolmoHandler(BaseVLMHandler):
    """Handler for Molmo models (allenai/Molmo-*)."""

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        kwargs["trust_remote_code"] = True
        kwargs["device_map"] = _get_device_map(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        from transformers import GenerationConfig

        inputs = self.processor.process(images=[image], text=prompt)
        model_dtype = self.torch_dtype or torch.bfloat16
        inputs = {
            k: v.to(device=self.model.device, dtype=model_dtype if v.is_floating_point() else None).unsqueeze(0)
            for k, v in inputs.items()
        }

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=["<|endoftext|>"],
            do_sample=do_sample,
        )
        if do_sample:
            gen_config.temperature = float(temperature)

        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs, gen_config, tokenizer=self.processor.tokenizer
            )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


@register_handler("granite-vision-3", "ibm-granite/granite-vision-3")
class GraniteVisionHandler(BaseVLMHandler):
    """Handler for IBM Granite Vision models (ibm-granite/granite-vision-*)."""

    def load(self) -> None:
        kwargs = {}
        kwargs["torch_dtype"] = torch.bfloat16

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)
        kwargs["low_cpu_mem_usage"] = True

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


@register_handler("granite-vision-4", "ibm-granite/granite-vision-4")
class GraniteVision41Handler(BaseVLMHandler):
    """Handler for IBM Granite Vision 4.x (e.g. granite-vision-4.1-4b).

    Custom architecture (SigLIP2 + Window Q-Former + DeepStack injection); uses
    AutoModelForImageTextToText. Requires trust_remote_code=True on
    transformers < 5.8.0, and processor.tokenizer.padding_side = "left" for
    correct batched generation.
    """

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        kwargs["trust_remote_code"] = True
        kwargs["device_map"] = _get_device_map(self.device)
        kwargs["low_cpu_mem_usage"] = True

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        # left padding is required for correct batched generation per IBM docs
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        except Exception:
            # fallback: format text, pass image separately
            text = prompt
            inputs = self.processor(
                text=text, images=[image], return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


class GenericHandler(BaseVLMHandler):
    """
    Generic fallback handler using AutoModelForVision2Seq + AutoProcessor.
    Falls back to pipeline if direct loading fails.
    """

    def load(self) -> None:
        kwargs = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype

        attn_impl = self._get_attn_impl()
        if attn_impl:
            kwargs["_attn_implementation"] = attn_impl

        kwargs["device_map"] = _get_device_map(self.device)
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, **kwargs
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self._use_pipeline = False
        except Exception as e:
            print(f"Direct loading failed ({e}), falling back to pipeline...")
            self._use_pipeline = True
            pipe_kwargs = {}
            if self.torch_dtype is not None:
                pipe_kwargs["torch_dtype"] = self.torch_dtype
            device = self.device if self.device >= 0 else -1
            self._pipeline = pipeline(
                "image-text-to-text",
                model=self.model_name,
                device=device,
                trust_remote_code=self.trust_remote_code,
                **pipe_kwargs,
            )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        gen_kwargs = self._get_gen_kwargs(max_new_tokens, temperature, do_sample)

        if self._use_pipeline:
            return self._generate_with_pipeline(image, prompt, gen_kwargs)
        else:
            return self._generate_direct(image, prompt, gen_kwargs)

    def _generate_direct(self, image, prompt, gen_kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            text_prompt = prompt

        inputs = self.processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _generate_with_pipeline(self, image, prompt, gen_kwargs):
        attempts = [
            lambda: self._pipeline(images=image, text=prompt, generate_kwargs=gen_kwargs),
            lambda: self._pipeline(image, prompt, generate_kwargs=gen_kwargs),
            lambda: self._pipeline(image, prompt=prompt, generate_kwargs=gen_kwargs),
        ]

        last_error = None
        for attempt in attempts:
            try:
                out = attempt()
                return self._extract_text_from_pipeline_output(out)
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Pipeline call failed. Last error: {last_error}")

    @staticmethod
    def _extract_text_from_pipeline_output(out: Any) -> str:
        if out is None:
            return ""
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, dict):
            if "generated_text" in out:
                return str(out["generated_text"]).strip()
            if "text" in out:
                return str(out["text"]).strip()
            return str(out).strip()
        if isinstance(out, (list, tuple)) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    return str(first["generated_text"]).strip()
                if "text" in first:
                    return str(first["text"]).strip()
            return str(first).strip()
        return str(out).strip()


# -----------------------------
# RITS API Handler
# -----------------------------

class RitsAPIHandler:
    """VLM handler that calls an OpenAI-compatible inference API (RITS or LiteLLM)."""

    RITS_MODEL_MAP = {
        'llama-3-3-70b':         'meta-llama/llama-3-3-70b-instruct',
        'pixtral-large':         'mistralai/Pixtral-Large-Instruct-2411',
        'qwen2.5-vl-72b':        'Qwen/Qwen2.5-VL-72B-Instruct',
        'deepseek-vl2':          'deepseek-ai/deepseek-vl2',
        'qwen3-vl-235b':         'Qwen/Qwen3-VL-235B-A22B-Instruct',
        'qwen3-vl-235b-think':   'Qwen/Qwen3-VL-235B-A22B-Thinking',
        'internvl3.5-241b':      'OpenGVLab/InternVL3_5-241B-A28B',
        'gpt-5.5':              'azure/gpt-5.5',
    }

    def __init__(self, model_name: str, api_base: str, api_key: Optional[str] = None, auth_style: str = "rits"):
        self.model_id = self.RITS_MODEL_MAP.get(model_name, model_name)
        self.api_base = api_base.rstrip('/')
        self.auth_style = auth_style
        if auth_style == "bearer":
            self.api_key = api_key or os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
        else:
            self.api_key = api_key or os.environ.get('RITS_API_KEY', '')

    def load(self) -> None:
        """No-op: API is stateless."""
        pass

    def format_prompt(self, instruction: str, question: str) -> str:
        return f"{instruction}\n{question}" if instruction else question

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs,
    ) -> str:
        import base64
        import io
        import requests

        buf = io.BytesIO()
        image.copy().convert('RGB').save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }
            ],
            "max_tokens": max_new_tokens,
        }
        if temperature != 0.0:
            payload["temperature"] = temperature

        if self.auth_style == "bearer":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "RITS_API_KEY": self.api_key,
            }

        resp = requests.post(
            f"{self.api_base}/v1/chat/completions",
            headers=headers,
            json=payload,
            verify=False,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# -----------------------------
# VLM Loading
# -----------------------------

def load_vlm(
    model_name: str,
    device: int,
    dtype: str,
    trust_remote_code: bool = False,
    use_flash_attn: bool = False,
    use_lora: bool = False,
    lora_path: str = None,
) -> BaseVLMHandler:
    """Load a VLM using the appropriate handler."""
    handler_cls = get_handler_class(model_name)
    print(f"Selected handler: {handler_cls.__name__}")

    handler = handler_cls(
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        use_flash_attn=use_flash_attn,
        use_lora=use_lora,
        lora_path=lora_path,
    )
    handler.load()
    return handler
