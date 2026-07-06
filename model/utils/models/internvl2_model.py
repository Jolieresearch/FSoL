from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class InternVL2Model(BaseModel):
    def __init__(self, model_id="OpenGVLab/InternVL2_5-8B-MPO", image_size=448):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='cuda'
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        

        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")
        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        

        self.device = next(self.model.parameters()).device

        self.transform = self.build_transform(image_size)
    
    @staticmethod
    def model_list():
        return [""]
    
    def build_transform(input_size=448):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=1)
        return 1 if "1" in response else 0

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 256) -> str:
        pixel_values = self.transform(image).unsqueeze(0).to(torch.bfloat16).to(self.device)

        generation_config = dict(max_new_tokens=max_tokens, do_sample=True)

        return self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config
        )
    
    def chat_multi_img(self, prompt: str, images: list[Image.Image], max_tokens: int = 1024) -> str:
        """
        Process multiple images with a prompt.
        For InternVL2, we concatenate multiple images.
        """

        pixel_values = torch.stack([self.transform(img) for img in images]).to(torch.bfloat16).to(self.device)
        
        generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
        
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config
        )
        
        return response
    
    def chat_multi_img_label(self, prompt: str, images: list[Image.Image]) -> tuple:
        """
        Multi-image labeling function that returns both prediction and probabilities from logits.
        Returns (prediction_text, probs) where probs is [prob_0, prob_1].
        """

        pixel_values = torch.stack([self.transform(img) for img in images]).to(torch.bfloat16).to(self.device)
        

        question = '<image>\n' * len(images) + prompt
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config=dict(max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True),
            history=None,
            return_history=True
        )
        


        from loguru import logger
        

        template = f'<|im_start|>User\n{question}<|im_end|><|im_start|>Assistant\n'
        inputs = self.tokenizer(template, return_tensors='pt').to(self.device)
        

        output_dict = self.model.generate(
            **inputs,
            pixel_values=pixel_values,
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        output_ids = output_dict.sequences
        scores = output_dict.scores
        

        generated_ids_for_check = output_ids[0][len(inputs.input_ids[0]):]
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids_for_check)
        

        target_pos = -1
        for i, token in enumerate(generated_tokens):
            token_str = str(token).strip().replace('▁', '')
            if token_str in ['0', '1']:
                target_pos = i
                break
        

        if target_pos >= 0 and target_pos < len(scores):
            logits = scores[target_pos][0, :]
        else:

            logits = scores[0][0, :] if len(scores) > 0 else None
        
        if logits is None:


            output_text = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            logger.warning(f"Could not extract logits, output: {output_text[:100]}")

        
        logits = logits
        

        logits_0 = logits[self.zero_id].unsqueeze(-1)
        logits_1 = logits[self.one_id].unsqueeze(-1)
        cls_logits = torch.cat([logits_0, logits_1], dim=-1)
        

        probs = torch.softmax(cls_logits, dim=-1)
        probs = probs.cpu().float().numpy().tolist()
        

        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        logger.debug(f"InternVL2 label output: {output_text}, probs: {probs}")
        
        return output_text, probs
