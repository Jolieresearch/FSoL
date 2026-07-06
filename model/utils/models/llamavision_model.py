from .base_model import BaseModel
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image


class LlamaVisionModel(BaseModel):
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", params={}):
        super().__init__()
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.params = {k: v for k, v in (params or {}).items() if k != 'use_api'}

    @staticmethod
    def model_list():
        return ["meta-llama/Llama-3.2-11B-Vision-Instruct"]
    
    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=30)
        if "0" in response:
            return 0
        elif "1" in response:
            return 1
        elif "not hateful" in response:
            return 0
        elif "hateful" in response:
            return 1
        else:
            return 1

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 256) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
       
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        print(output_text)
        
        return output_text[0]