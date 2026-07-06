from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import einops
from loguru import logger
import torch
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info


class TextModel(BaseModel):
    def __init__(self, model_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", params={}):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,

            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.params = {k: v for k, v in (params or {}).items() if k != 'use_api'}
        
    @staticmethod
    def model_list():
        return ["Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", "Qwen/Qwen2.5-72B-Instruct-AWQ"]
    
    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        output = self.chat_text_batch([prompt], max_tokens)
        return output[0]
    
    def chat_text_batch(self, prompts: list[str], max_tokens: int = 2048) -> list[str]:
        conversations = [
            [{
                "role": "user", 
                "content": prompt
            }] for prompt in prompts
        ]
        
        texts = self.tokenizer.apply_chat_template(  
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text
    