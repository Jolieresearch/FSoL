from .base_model import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import requests
from loguru import logger


class LlavaModel(BaseModel):
    """
    Simplified version of Llava. Provides a single chat_predict interface:
      chat_predict(prompt: str, image: Image.Image) -> str

    Where:
      - prompt: User input text
      - image: A single PIL.Image
      - The return value is the model's response to the image and prompt.
    """
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", para: dict = {}):
        super().__init__()
        

        load_in_4bit = para.get('load_in_4bit', False)
        load_in_8bit = para.get('load_in_8bit', True)
        
        if load_in_4bit:
            logger.info("Using 4-bit quantization for extreme memory savings")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.warning("No quantization specified, using float16 (high memory usage)")
            quantization_config = None
        
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_kwargs
        )
        

        if para.get('use_gradient_checkpointing', False):
            logger.info("Gradient checkpointing enabled")
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        
        self.processor = AutoProcessor.from_pretrained(model_id)
        

        self.zero_id = self.processor.tokenizer.convert_tokens_to_ids("0")
        self.one_id = self.processor.tokenizer.convert_tokens_to_ids("1")
        

        self.device = next(self.model.parameters()).device
        

        self.device = next(self.model.parameters()).device
        
    @staticmethod
    def model_list():
        return ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-34b-hf"]

    def chat_predict(self, prompt: str, image: Image.Image) -> int:
        response = self.chat_img(prompt, image, max_tokens=10)

        response = response[0]
        return 1 if "1" in response else 0

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 1024) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors='pt'
        ).to(self.device, torch.float16)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True

        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text[0]
    
    def chat_multi_img(self, prompt: str, images: list[Image.Image], max_tokens: int = 1024) -> str:
        """
        Process multiple images with a prompt.
        For LLava models, we create a conversation with multiple image entries.
        """

        content = [{"type": "text", "text": prompt}]
        for _ in images:
            content.append({"type": "image"})
        
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=images,
            text=formatted_prompt,
            return_tensors='pt'
        ).to(self.device, torch.float16)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True
        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(f"LLava multi-img output: {output_text}")
        
        return output_text[0]
    
    def chat_multi_img_label(self, prompt: str, images: list[Image.Image]) -> tuple:
        """
        Multi-image labeling function that returns both prediction and probabilities from logits.
        Returns (prediction_int, probs) where probs is [prob_0, prob_1].
        Now returns integer prediction directly.
        """

        content = [{"type": "text", "text": prompt}]
        for _ in images:
            content.append({"type": "image"})
        
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=images,
            text=formatted_prompt,
            return_tensors='pt'
        ).to(self.device, torch.float16)

        output_dict = self.model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            temperature=None
        )
        
        output_ids = output_dict.sequences
        scores = output_dict.scores
        

        generated_ids_for_check = output_ids[0][len(inputs.input_ids[0]):]
        output_text_full = self.processor.tokenizer.decode(generated_ids_for_check, skip_special_tokens=True)
        
        logger.debug(f"Generated text: {output_text_full[:200]}")
        

        prediction = None
        probs = None
        
        if len(scores) > 0:
            logits = scores[0][0, :]
            

            logits_0 = logits[self.zero_id]
            logits_1 = logits[self.one_id]
            
            logger.debug(f"First token - Logit for '0': {logits_0.item():.4f}, Logit for '1': {logits_1.item():.4f}")
            

            logit_diff = abs(logits_1.item() - logits_0.item())
            if logit_diff > 1.0:
                cls_logits = torch.stack([logits_0, logits_1])
                probs_tensor = torch.softmax(cls_logits, dim=-1)
                probs = probs_tensor.cpu().numpy().tolist()
                prediction = 1 if probs[1] > probs[0] else 0
                logger.debug(f"Using logits-based prediction: {prediction}, probs: {probs}")
        

        if prediction is None:

            import re
            first_digit = re.search(r'\b[01]\b', output_text_full)
            if first_digit:
                prediction = int(first_digit.group())
                probs = [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
                logger.debug(f"Using text-based prediction: {prediction}")
            else:

                text_lower = output_text_full.lower()
                if 'fake' in text_lower or '1' in output_text_full:
                    prediction = 1
                    probs = [0.3, 0.7]
                elif 'real' in text_lower or '0' in output_text_full:
                    prediction = 0
                    probs = [0.7, 0.3]
                else:
                    logger.warning(f"Cannot parse prediction from: {output_text_full[:100]}")
                    prediction = 1
                    probs = [0.4, 0.6]
        
        return prediction, probs