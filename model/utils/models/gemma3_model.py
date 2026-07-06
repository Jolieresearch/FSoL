import base64
from io import BytesIO

import torch
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)

from .base_model import BaseModel


def resize_image(image: Image.Image, max_size=640):
    """
    Resize an image while maintaining its aspect ratio, ensuring the longest side is max_size pixels.

    Args:
        image (PIL.Image.Image): The image to resize
        max_size (int): Maximum length of the longest dimension

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


class Gemma3Model(BaseModel):
    def __init__(self, model_id="google/gemma-3-27b-it", params={}):
        super().__init__()
        

        load_params = {
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
        }
        

        if params.get('load_in_4bit'):
            from transformers import BitsAndBytesConfig
            load_params['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Loading Gemma3 with 4-bit quantization (~75% memory reduction)")
        elif params.get('load_in_8bit'):
            from transformers import BitsAndBytesConfig
            load_params['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            logger.info("Loading Gemma3 with 8-bit quantization (~50% memory reduction)")
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **load_params)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        

        excluded_keys = {'use_api', 'load_in_4bit', 'load_in_8bit'}
        self.params = {k: v for k, v in (params or {}).items() if k not in excluded_keys}

        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")

        self.fake_id = self.tokenizer.convert_tokens_to_ids("Ġfake")
        self.real_id = self.tokenizer.convert_tokens_to_ids("Ġreal")

    @staticmethod
    def model_list():
        return ["google/gemma-3-27b-it", "google/gemma-3-4b-it"]

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

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)

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

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)

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
        

        strict_prompt = f"{prompt}\n\nYou MUST respond with ONLY a single digit: either 0 or 1. Do not write any other words, explanations, or text."
        conversation[0]["content"].append({"type": "text", "text": strict_prompt})
        
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

        return output_text[0], probs
