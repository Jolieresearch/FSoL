import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from .base_model import BaseModel


class InternVL3Model(BaseModel):
    def __init__(self, model_id="OpenGVLab/InternVL3_5-38B-HF", params={}):
        super().__init__()
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="cuda", torch_dtype=torch.bfloat16, quantization_config=quant_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.params = {k: v for k, v in (params or {}).items() if k != 'use_api'}

    @staticmethod
    def model_list():
        return ["OpenGVLab/InternVL3_5-38B-HF"]

    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=30)
        if "0" in response:
            return 0
        elif "1" in response:
            return 1
        elif "not fake" in response:
            return 0
        elif "fake" in response:
            return 1
        else:
            return 1

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 256) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        print(decoded_output)

        return decoded_output
