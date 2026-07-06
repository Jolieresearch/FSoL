import base64
from io import BytesIO

import torch
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from .base_model import BaseModel


def resize_image(image: Image.Image, max_size=384):
    """
    Resize an image while maintaining its aspect ratio, ensuring the longest side is max_size pixels.

    Args:
        image (PIL.Image.Image): The image to resize
        max_size (int): Maximum length of the longest dimension (default: 384 for memory optimization)

    Returns:
        PIL.Image.Image: Resized image
    """

    width, height = image.size


    if width >= height:

        scaling_factor = max_size / width
    else:

        scaling_factor = max_size / height


    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)


    return image.resize((new_width, new_height), Image.LANCZOS)


def pil_image_to_base64(image, image_format="PNG"):
    """
    Convert a PIL.Image.Image object to a Base64 string.

    Args:
        image (PIL.Image.Image): The image to be converted.
        image_format (str): Format to save the image, e.g., "PNG", "JPEG".

    Returns:
        str: The Base64-encoded string of the image.
    """

    image = resize_image(image)

    buffer = BytesIO()


    image.save(buffer, format=image_format)


    buffer.seek(0)


    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")


    buffer.close()

    return image_base64


class Qwen3VLModel(BaseModel):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", params={}):
        super().__init__()
        

        logger.info("Loading model with memory optimization...")
        

        try:
            import flash_attn
            use_flash_attention = True
            logger.info("Flash Attention 2 is available, will use it for memory optimization")
        except ImportError:
            use_flash_attention = False
            logger.warning("Flash Attention 2 not available, using standard attention")
        

        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            
            if use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            logger.info("Loading model with 8-bit quantization...")
            
        except ImportError:
            logger.warning("BitsAndBytes not available, loading with bfloat16...")
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        

        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.params = {k: v for k, v in (params or {}).items() if k != 'use_api'}

        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")

        self.harmful_id = self.tokenizer.convert_tokens_to_ids("Ġfake")
        self.harmless_id = self.tokenizer.convert_tokens_to_ids("Ġreal")
        
        logger.info(f"Model loaded successfully with device_map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'N/A'}")

    @staticmethod
    def model_list():
        return [
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct"
        ]

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Chat with text-only input (no images)
        Used for tasks like insight management that only need text processing
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)


        inputs = self.processor(text=[text_prompt], padding=True, return_tensors="pt").to(
            self.model.device
        )


        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        

        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        return output_text[0]

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 512) -> str:
        image = resize_image(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
            self.model.device
        )


        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        

        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        return output_text[0]

    def chat_img_batch(
        self, prompts: list[str], images: list[Image.Image], max_tokens: int = 256
    ) -> list[str]:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for prompt in prompts
        ]

        text_prompts = [
            self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            for conversation in conversations
        ]

        inputs = self.processor(text=text_prompts, images=images, padding=True, return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [output_ids[i][len(inputs.input_ids[i]) :] for i in range(len(output_ids))]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)

        return output_text

    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=512)
        response = response if isinstance(response, list) else response
        return 1 if "1" in response else 0

    def chat_multi_img_predict(self, prompt: str, images: list[Image.Image]) -> int:
        """
        Multi-image prediction function for predicting with multiple images.
        Returns 1 or 0 based on the model's response.
        """
        response = self.chat_multi_img(prompt, images)
        response = response if isinstance(response, list) else response
        return 1 if "1" in response else 0
































































    def chat_label(self, prompt: str, image: Image.Image) -> int:
        image = resize_image(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
            self.model.device
        )


        with torch.no_grad():
            output_dict = self.model.generate(
                **inputs,
                max_new_tokens=5,
                **self.params,
                return_dict_in_generate=True,
                output_logits=True,
                return_legacy_cache=True,
            )
        
        output_ids = output_dict.sequences
        logits = output_dict.logits



        first_token_logits = logits[0][0, :]

        logits_1 = first_token_logits[self.one_id].unsqueeze(-1)
        logits_0 = first_token_logits[self.zero_id].unsqueeze(-1)
        cls_logits = torch.cat([logits_0, logits_1], dim=-1)

        probs = torch.softmax(cls_logits, dim=-1)
        probs = probs.cpu().numpy()

        probs = probs.tolist()

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(f"output_text: {output_text}")
        logger.debug(f"probs: {probs}")
        

        del inputs, output_dict, output_ids, logits, cls_logits, generated_ids
        torch.cuda.empty_cache()

        return output_text[0], probs

    def chat_multi_img(self, prompt: str, images: list[Image.Image]) -> str:
        conversation = [
            {
                "role": "user",
                "content": [],
            }
        ]
        for i in range(len(images)):
            conversation[0]["content"].append(
                {"type": "image", "image": f"data:image;base64,{pil_image_to_base64(images[i])}"}
            )
        conversation[0]["content"].append({"type": "text", "text": prompt})
        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        

        processor_kwargs = {"text": [text_prompt], "padding": True, "return_tensors": "pt"}
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
            
        inputs = self.processor(**processor_kwargs).to(self.model.device)

        output_ids = self.model.generate(**inputs, **self.params, max_new_tokens=1024)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)

        return output_text[0]

    def chat_multi_img_label(self, prompt: str, images: list[Image.Image]) -> tuple:
        """
        Multi-image labeling function similar to chat_label but for multiple images.
        Returns prediction and probabilities.
        """
        conversation = [
            {
                "role": "user",
                "content": [],
            }
        ]
        for i in range(len(images)):
            conversation[0]["content"].append(
                {"type": "image", "image": f"data:image;base64,{pil_image_to_base64(images[i])}"}
            )
        conversation[0]["content"].append({"type": "text", "text": prompt})
        
        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        

        processor_kwargs = {"text": [text_prompt], "padding": True, "return_tensors": "pt"}
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
            
        inputs = self.processor(**processor_kwargs).to(self.model.device)


        with torch.no_grad():
            output_dict = self.model.generate(
                **inputs,
                max_new_tokens=1,
                **self.params,
                return_dict_in_generate=True,
                output_logits=True,
                return_legacy_cache=True,
            )
        
        output_ids = output_dict.sequences
        logits = output_dict.logits


        logits = logits[0][0, :]

        logits_1 = logits[self.one_id].unsqueeze(-1)
        logits_0 = logits[self.zero_id].unsqueeze(-1)
        cls_logits = torch.cat([logits_0, logits_1], dim=-1)

        probs = torch.softmax(cls_logits, dim=-1)
        probs = probs.cpu().numpy()

        probs = probs.tolist()

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(f"output_text: {output_text}")
        logger.debug(f"probs: {probs}")
        

        del inputs, output_dict, output_ids, logits, cls_logits, generated_ids
        if image_inputs is not None:
            del image_inputs
        if video_inputs is not None:
            del video_inputs
        torch.cuda.empty_cache()

        return output_text[0], probs
