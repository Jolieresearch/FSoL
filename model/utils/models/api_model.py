from PIL import Image
import base64
from io import BytesIO
import requests
import os
from typing import Optional
import dotenv
import time
from loguru import logger




class APIModel():
    def __init__(self, model_id: str, params={}):
        """
        Initialize APIModel with model_id and API key.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4-vision-preview")
        """
        super().__init__()
        dotenv.load_dotenv()
        self.model_id = model_id
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable is not set")
        
        self.api_base = os.getenv("API_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.params = params

    @staticmethod
    def model_list():
        return ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "Pro/Qwen/Qwen2.5-VL-7B-Instruct", "meta-llama/llama-3.2-11b-vision-instruct",
                "google/gemma-3-4b-it", "google/gemma-3-12b-it", "microsoft/phi-4-multimodal-instruct", "openai/gpt-4o-2024-11-20", "openai/gpt-4o-mini-2024-07-18"
                ]
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        image = image.convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')




        



            






    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [image], max_tokens)

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [], max_tokens)

    def chat_multi_img(self, prompt: str, images: list[Image.Image], max_tokens: int = 2048) -> str:
        """
        Generate text response for the given prompt and multiple images using OpenAI API.
        
        Args:
            prompt: Text prompt to process
            images: List of images to analyze (list of PIL Image objects)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        content = [{"type": "text", "text": prompt}]
        

        for image in images:
            base64_image = self._encode_image(image)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            if 'Qwen' in self.model_id:
                image_content["image_url"]["detail"] = 'high'
            if 'gpt' in self.model_id:
                image_content["image_url"]["detail"] = 'low'
            content.append(image_content)

        payload = {
            "model": self.model_id,
            "messages": [









                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.000001
        }

        payload.update(self.params)
        

        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                

                if response.status_code == 429 and attempt < max_retries - 1:
                    retry_delay = 30 * (2 ** attempt)
                    logger.warning(f"Rate limit (429) hit. Retrying after {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                response.raise_for_status()
                
                result = response.json()

                num_images = len([c for c in payload['messages'][0]['content'] if c.get('type') == 'image_url'])
                logger.debug(f"API request: {num_images} images + text: {payload['messages'][0]['content'][0]['text'][:100]}...\n Response: {result}")
                return result['choices'][0]['message']['content'].strip()
                
            except requests.exceptions.RequestException as e:

                time.sleep(30)
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")

def test_multi_image_chat(image_paths: list[str], prompt: str = "What's in these images?", model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct") -> str:
    """
    Test function to verify if multiple images can be used for chat with the model.
    
    Args:
        image_paths: List of paths to the image files
        prompt: Text prompt to send with the images
        model_id: Model identifier to use for the test
        
    Returns:
        str: Response from the model
    """
    try:

        images = [Image.open(path).convert('RGB') for path in image_paths]
        

        api_model = APIModel(model_id=model_id)
        

        response = api_model.chat_multi_img(prompt, images)
        
        return response
    except Exception as e:
        return f"Error during test: {str(e)}"

if __name__ == "__main__":
    import sys


    if len(sys.argv) > 2:
        image_path1 = sys.argv[1]
        image_path2 = sys.argv[2]

        prompt = sys.argv[3] if len(sys.argv) > 3 else "Compare these two images. What can you see in them?"
    else:

        image_path1 = "sketch/SCR-20250328-omdk.png"
        image_path2 = "sketch/SCR-20250328-phzc.png"
        prompt = "Compare these two images. What can you see in them?"
    
    image_paths = [image_path1, image_path2]
    
    print(f"Testing multi-image chat with images: {image_paths}")
    print(f"Prompt: {prompt}")
    

    response = test_multi_image_chat(image_paths, prompt)
    
    print("\nResponse from model:")
    print(response)

