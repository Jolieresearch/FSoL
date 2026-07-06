from typing import Dict

from .models.base_model import BaseModel
from .models.llamavision_model import LlamaVisionModel
from .models.internvl2_model import InternVL2Model
from .models.internvl3_model import InternVL3Model
from .models.qwen2vl_model import Qwen2VLModel
from .models.qwen3vl_model import Qwen3VLModel
from .models.llava_model import LlavaModel
from .models.gemma3_model import Gemma3Model
from .models.text_model import TextModel
from .models.pipeline_model import PipelineModel
from .models.api_model import APIModel

def MLLMFactory(model_id: str, params=None|Dict):
    """
    Factory function to create model instances.
    
    Args:
        model_id: Model identifier
        params: Model parameters dict, can include 'use_api' flag
        
    Returns:
        Model instance
    """
    if params is None:
        params = {}
    

    use_api = params.get('use_api', False)
    
    if use_api:

        return APIModel(model_id, params)
    

    model_classes = [LlamaVisionModel, InternVL2Model, InternVL3Model, Qwen2VLModel, Qwen3VLModel, LlavaModel, Gemma3Model, TextModel, PipelineModel, APIModel]
    for model_class in model_classes:
        if model_id in model_class.model_list():
            return model_class(model_id, params)
    raise ValueError(f"Model {model_id} not supported")

