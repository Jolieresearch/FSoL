from .base_model import BaseModel

from PIL import Image
import requests
import einops
from loguru import logger
import torch
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
from lmdeploy import pipeline, TurbomindEngineConfig


def pil_image_to_base64(image, image_format="PNG"):
    """
    Convert a PIL.Image.Image object to a Base64 string.

    Args:
        image (PIL.Image.Image): The image to be converted.
        image_format (str): Format to save the image, e.g., "PNG", "JPEG".

    Returns:
        str: The Base64-encoded string of the image.
    """

    buffer = BytesIO()


    image.save(buffer, format=image_format)


    buffer.seek(0)


    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")


    buffer.close()

    return image_base64


class PipelineModel(BaseModel):
    def __init__(self, model_id="OpenGVLab/InternVL2_5-38B-MPO-AWQ", params={}):
        super().__init__()
        self.pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192, tp=1, cache_max_entry_count=0.2), device_map="cuda")

        self.params = {k: v for k, v in (params or {}).items() if k != 'use_api'}
        


        
    @staticmethod
    def model_list():
        return ["OpenGVLab/InternVL2_5-38B-MPO-AWQ"]
    
    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 512) -> str:
        response = self.pipe((prompt, image))
        logger.debug(f'response: {response}')
        return response.text
    

        conversations = [
            [{
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }] for prompt in prompts
        ]
        
        text_prompts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            ) for conversation in conversations
        ]
        
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text
    
    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=512)
        response = response if isinstance(response, list) else response
        return 1 if "1" in response else 0
    

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_dict = self.model.generate(
            **inputs,
            max_new_tokens=1,
            **self.params,
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=True
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
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(f"output_text: {output_text}")
        logger.debug(f"probs: {probs}")
        
        return output_text[0], probs
    
    def chat_multi_img(self, prompt: str, images: list[Image.Image]) -> str:
        response = self.pipe((prompt, images))
        logger.debug(f"response: {response}")
        return response.text