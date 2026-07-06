import re
from ..utils.mllm_factory import MLLMFactory
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import Dict
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import os
import json
import multiprocessing
from functools import partial
from omegaconf import OmegaConf


prompt_template = """
Now, a new experience arrives containing the analysis of one fake and one authentic video, a summary of the differences between the two categories, and a detection rule for this type of fake news.
Your task is to distill new insights from the experience and update the existing insights by choosing exactly one operation:

[
  {{
    "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
    "target": "<index/none>",
    "insight": "<new/revised text>"
  }}
]

### STRICT RULES:
1. ADD only if:
   - add new insights that are very different from existing insights and relevant for other fake news detection.
2. EDIT must:
   - if any existing insight is not general enough or can be enhanced, rewrite and improve it.
3. UPVOTE if:
   - if the existing insight is strongly relevant for current insight
4. DOWNVOTE if:
   - if one existing insight is contradictory or similar/duplicated to other existing insight. 
5. MAXIMUM {size} insights preserved
6. OUTPUT ONLY VALID JSON

### CONTEXT:
Current Insights Set (importance order):
{cur_set_str}

New Coming Experience:
{new_insight}

### PROCESSING STEPS:
1. Ensure the added and edited insights are concise, clear while keeping them 2 or 3 sentences.
2. Ensure the insights are concise and easy to follow.
3. Actively downvote insights that are vague or hard to understand, and maintain the insight set at {size} items.
4. If UPVOTE/DOWNVOTE is used in an operation, include only 'none' in the insight of the operation.
5. Try to make every insight useful, make more upvotes, and downvotes.
6. Return only the JSON operations, no additional text.
"""

class InsightSetManager:
    def __init__(self, size: int):
        self.size = size
        self.set = []

        self.set.append({
            "insight": "placeholder",
            "importance": 2
        })
      
    def init_from_df(self, df: pd.DataFrame):

        self.set = df.to_dict(orient='records')

    def extract_instruction(self, json_str: str):
        """
        Expects a JSON string of operations in the form:
        [
          {
            "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
            "target": <index of target if applicable>,
            "insight": "<new or updated insight text if applicable>"
          }
        ]
        """
        pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        match = pattern.search(json_str)
        if match:
            json_str = match.group(1).strip()
        try:
            operations = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_str}")
            return

        if isinstance(operations, dict):
            operations = [operations]
      
        for operation in operations:
            op_type = operation.get("operation", "").upper()
            target = operation.get("target", None)
            insight_text = operation.get("insight", None)
            

            if isinstance(target, str) and target != "" and target != 'none':
                try:
                    target = int(target)
                except ValueError as e:

                    if op_type != 'ADD':
                        logger.error(f"Failed to convert target to int: {target}")
                        logger.error(f"Error: {e}")
                        continue
          
            if op_type == "ADD":
                self._add(insight_text)
            elif op_type == "EDIT":
                self._edit(target, insight_text)
            elif op_type == "UPVOTE":
                self._upvote(target)
            elif op_type == "DOWNVOTE":
                self._downvote(target)
            else:
                logger.error(f"Unknown operation: {op_type}")


        self.set = [item for item in self.set if item["importance"] > 0]
        self.set.sort(key=lambda x: x["importance"], reverse=True)
        

        if len(self.set) > self.size + 5:

            self.set.sort(key=lambda x: x["importance"], reverse=True)

            self.set = self.set[:self.size]

    def _add(self, insight: str):
        logger.debug(f"Adding new insight with importance=2: {insight}")
        new_item = {"insight": insight, "importance": 2}
        self.set.append(new_item)


    def _edit(self, index: int, new_insight: str):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Editing insight at index {index}, incrementing importance")
        self.set[index]["insight"] = new_insight
        self.set[index]["importance"] += 1

    def _upvote(self, index: int):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Upvoting insight at index {index}")
        self.set[index]["importance"] += 1

    def _downvote(self, index: int):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Downvoting insight at index {index}")

        if self.set[index]["importance"] < 9999:
            self.set[index]["importance"] -= 1

    def _enforce_size(self):

        while len(self.set) > self.size:

            if self.set[0]["importance"] == 9999:
                logger.debug("Removing next oldest insight (placeholder is kept).")
                self.set.pop(1)
            else:
                logger.debug("Removing oldest insight.")
                self.set.pop(0)

    def save_to_df(self, path: str):
        df = pd.DataFrame(self.set)
        df.to_json(path, lines=True, index=False, orient='records')

    def get_cur_set_str(self):

        return "\n".join(
            f"{i}: {item['insight']} (importance: {item['importance']})"
            for i, item in enumerate(self.set)
        )

    def export_data(self):
        """
        Export the internal set data as a serializable structure.
        
        Returns:
            dict: A serializable representation of the insight set
        """
        return {
            'size': self.size,
            'set': self.set.copy()
        }
    
    def import_data(self, data):
        """
        Import data to recreate the insight set.
        
        Args:
            data (dict): Data structure containing 'size' and 'set' keys
        """
        if 'size' in data:
            self.size = data['size']
        if 'set' in data:
            self.set = data['set']

        self.set.sort(key=lambda x: x["importance"], reverse=True)

class Manage_Runner:
    def __init__(self, cfg: DictConfig, **kwargs):
        """
        Initialize Manage Runner for fake video datasets
        
        Args:
            model_id: ID of the model to use
        """
        self.model_id = cfg.model_id
        self.model_params = cfg.para if cfg.para else {}
        self.model = MLLMFactory(self.model_id, self.model_params)
        self.dataset = cfg.dataset
        self.result = None
        
        self.insight_name = cfg.insight_name
        self.reference_name = cfg.reference_name
        self.data_split = cfg.get('data_split', 'train')
        


        insight_dir = 'insight/test' if self.data_split == 'test' else 'insight'
        insight_data = []
        with open(f'data/{self.dataset}/{insight_dir}/{self.insight_name}.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    insight_data.append(item)
        self.insight_df = pd.DataFrame(insight_data)
        self.insight_df['id1'] = self.insight_df['id1'].astype(str)
        self.insight_df['id2'] = self.insight_df['id2'].astype(str)
        

        reference_dir = 'reference/test' if self.data_split == 'test' else 'reference'
        self.output_path = f'data/{self.dataset}/{reference_dir}/{self.reference_name}.jsonl'
        self.insight_set_manager = InsightSetManager(size=cfg.size)
        

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        

        self.save_df = pd.DataFrame()
        
        self.start = OmegaConf.select(cfg, "start", default=0)
        

    def run(self, num_processes=1):
        """
        Run inference on insights sequentially with a single model instance
        
        Args:
            num_processes: Number of parallel processes (default 1 for memory efficiency)
        
        Returns:
            None: Results are saved to the output file
        """

        insights_to_process = []
        for index, row in self.insight_df.iterrows():

            if index < self.start:
                continue
            insights_to_process.append((index, row))
        

        if not insights_to_process:
            logger.info("No insights to process.")
            return
        
        logger.info(f"Processing {len(insights_to_process)} insights using 1 process with synchronized insight set")
        

        results = []
        for idx, row in tqdm(insights_to_process, desc="Processing insights"):
            try:
                id1 = row['id1']
                id2 = row['id2']
                insight = row['insight']
                

                current_insight_set_str = self.insight_set_manager.get_cur_set_str()
                

                prompt = prompt_template.format(
                    size=self.insight_set_manager.size,
                    cur_set_str=current_insight_set_str,
                    new_insight=insight
                )
                

                instruction = self.model.chat_text(prompt)
                
                result = {
                    'id1': id1,
                    'id2': id2,
                    'insight': insight,
                    'instruction': instruction
                }
                

                self.insight_set_manager.extract_instruction(instruction)
                

                self.insight_set_manager.save_to_df(self.output_path)
                
                results.append(result)
                

                if len(results) % 5 == 0:
                    logger.info(f"Processed {len(results)} insights")
                    
            except Exception as e:
                logger.error(f"Error processing insight at index {idx}: {e}")
                continue
        

        self.insight_set_manager.save_to_df(self.output_path)
        logger.info(f"All insights processed successfully. Total: {len(results)}/{len(insights_to_process)}")

    def log_result(self):
        pass


def process_insight_with_sync(args):
    """
    Process a single insight with synchronized insight set updates
    using the exported/imported data approach
    
    Args:
        args: Tuple containing (index, row, model_id, model_params, size, 
                               shared_insights, lock, results, output_path)
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    idx, row, model_id, model_params, size, shared_insights, lock, results, output_path = args
    
    try:

        model = MLLMFactory(model_id, model_params)
        
        id1 = row['id1']
        id2 = row['id2']
        insight = row['insight']
        

        with lock:

            current_data = shared_insights['data']
        

        local_manager = InsightSetManager(size=size)
        local_manager.import_data(current_data)
        

        current_insight_set_str = local_manager.get_cur_set_str()
        

        prompt = prompt_template.format(
            size=size,
            cur_set_str=current_insight_set_str,
            new_insight=insight
        )
        

        instruction = model.chat_text(prompt)
        
        result = {
            'id1': id1,
            'id2': id2,
            'insight': insight,
            'instruction': instruction
        }
        

        with lock:

            updated_data = shared_insights['data']
            

            updated_manager = InsightSetManager(size=size)
            updated_manager.import_data(updated_data)
            

            updated_manager.extract_instruction(instruction)
            

            shared_insights['data'] = updated_manager.export_data()
            

            updated_manager.save_to_df(output_path)
            

            results.append(result)
            

            if len(results) % 5 == 0:
                logger.info(f"Processed {len(results)} insights")
        
        return True
    except Exception as e:
        logger.error(f"Error processing insight: {e}")
        return False