import numpy as np
from typing import Optional, Union
from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline, AutoProcessor, AutoModel
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import json
import multiprocessing
from functools import partial

from ..utils.mllm_factory import MLLMFactory
from ..utils.video_utils import load_video_frames, load_video_data


prompt_template = """
Now given the video (shown as individual frames), with the Text: {text} extracted from the video, your task is to determine whether this video contains fake/misleading information or not, in order to maintain the authenticity and integrity of information on the Internet.
Please leverage your extensive knowledge to deeply analyze and understand this video, and give your final judgment.
The insight set which can be referred to:
{insight_set}. \n
Your output should strictly follow the format: Thought: [Your analysis]\nAnswer: [fake/real].
"""


class InsightSetManager:
    def __init__(self, size: int, strategy: str = 'topk'):
        self.size = size
        self.set = []
        self.set.append({'insight': 'placeholder'})

        if strategy != 'topk':
            self.encoder = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=512, device='cuda')
            
        self.insight_text_embeddings = None
        self.insight_image_embeddings = None
        self.item_to_insights_map = {}
        
    def init_from_df(self, df: pd.DataFrame):

        self.set = df.to_dict(orient='records')
        
    def get_cur_set_str(self, tok_k):

        top_k_insights = self.set[:tok_k]
        

        return '\n'.join([f"{i}: {insight['insight']}" for i, insight in enumerate(top_k_insights)])
    
    def extract_image_features(self, image_path):
        """Extract features from an image using CLIP model"""
        try:

            image = Image.open(image_path).convert("RGB")

            inputs = self.image_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: val.to('cuda') for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.image_model(**inputs)
                image_features = outputs.image_embeds
            return image_features[0].cpu()
        except Exception as e:
            logger.error(f"Error extracting image features from {image_path}: {e}")
            return None
    
    def compute_insight_embeddings(self, insight_img_dir):
        """Compute and store embeddings for insights text and images"""
        logger.info("Computing embeddings for insights (text and images)")
        

        insight_texts = [item['insight'] for item in self.set]
        

        insight_images = []
        for item in self.set:
            if 'img' in item and item['img']:
                img_path = os.path.join(insight_img_dir, item['img'])
                insight_images.append(img_path)
            else:

                insight_images.append(None)
        

        self.insight_text_embeddings = self.encoder.encode(insight_texts, normalize_embeddings=True)
        

        valid_images = [img for img in insight_images if img is not None and os.path.exists(img)]
        valid_indices = [i for i, img in enumerate(insight_images) if img is not None and os.path.exists(img)]
        
        if valid_images:
            image_embeddings = self.encoder.encode(valid_images, normalize_embeddings=True)
            

            self.insight_image_embeddings = np.zeros((len(insight_texts), self.insight_text_embeddings.shape[1]))
            

            for idx, embedding_idx in enumerate(valid_indices):
                self.insight_image_embeddings[embedding_idx] = image_embeddings[idx]
        else:

            self.insight_image_embeddings = np.zeros((len(insight_texts), self.insight_text_embeddings.shape[1]))
            

        if not isinstance(self.insight_text_embeddings, torch.Tensor):
            self.insight_text_embeddings = torch.tensor(self.insight_text_embeddings)
        
        if not isinstance(self.insight_image_embeddings, torch.Tensor):
            self.insight_image_embeddings = torch.tensor(self.insight_image_embeddings)
        
    def compute_item_embeddings_and_topk_map(self, input_df, img_dir, insight_img_dir=None, top_k=5, alpha=0.2, dataset=None, num_frames=8, sample_strategy='uniform'):
        """
        Compute embeddings for all items and map them to their top-k insights
        considering both text and video visual similarities
        
        Args:
            input_df: DataFrame containing item data (with 'vid' field for videos)
            img_dir: Directory containing video frames (frames_16)
            insight_img_dir: Directory containing insight images
            top_k: Number of top insights to retrieve per item
            alpha: Weight for text vs visual similarity (alpha=0.2 means 20% text, 80% visual)
            dataset: Dataset name for loading video metadata
            num_frames: Number of frames to sample
            sample_strategy: Strategy for sampling frames ('uniform' or 'sequential')
            
        Returns:
            Dictionary mapping item IDs to their top-k insight indices
        """
        logger.info(f"Computing embeddings and top-{top_k} insights for all videos (text + visual)")
        

        if insight_img_dir is None:
            insight_img_dir = img_dir
        

        if self.insight_text_embeddings is None:
            self.compute_insight_embeddings(insight_img_dir)
            

        item_texts = []
        item_video_grids = []
        item_ids = []
        
        dataset_path = f"data/{dataset}" if dataset else None
        
        for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Preparing videos"):
            vid = str(row['vid'])
            

            try:
                video_frames = load_video_frames(img_dir, vid, max_frames=num_frames, sample_strategy=sample_strategy)
                

                if dataset_path:
                    video_data = load_video_data(dataset_path, vid)
                    text = video_data.get('text', '')
                else:
                    text = row.get('text', '')
                
                item_texts.append(text)
                item_video_grids.append(video_frames)
                item_ids.append(vid)
            except Exception as e:
                logger.warning(f"Failed to load video {vid}: {e}")
                continue
                

        item_text_embeddings = self.encoder.encode(item_texts, normalize_embeddings=True)
        


        all_frames = []
        frame_counts = []
        for frames in item_video_grids:
            all_frames.extend(frames)
            frame_counts.append(len(frames))
        

        all_frame_embeddings = self.encoder.encode(all_frames, normalize_embeddings=True)
        

        item_image_embeddings = []
        start_idx = 0
        for count in frame_counts:
            end_idx = start_idx + count
            video_embedding = np.mean(all_frame_embeddings[start_idx:end_idx], axis=0)
            item_image_embeddings.append(video_embedding)
            start_idx = end_idx
        
        item_image_embeddings = np.array(item_image_embeddings)
        

        if not isinstance(item_text_embeddings, torch.Tensor):
            item_text_embeddings = torch.tensor(item_text_embeddings)
        
        if not isinstance(item_image_embeddings, torch.Tensor):
            item_image_embeddings = torch.tensor(item_image_embeddings)

        logger.info(f"Item text embeddings dtype: {item_text_embeddings.dtype}")
        logger.info(f"Item image embeddings dtype: {item_image_embeddings.dtype}")
        
            

        text_similarity_scores = torch.matmul(item_text_embeddings, self.insight_text_embeddings.T)
        


        visual_similarity_scores = torch.matmul(item_image_embeddings.to(torch.float32), self.insight_image_embeddings.to(torch.float32).T)
        


        combined_similarity = alpha * text_similarity_scores + (1-alpha) * visual_similarity_scores
        

        item_to_insights = {}
        for i, item_id in enumerate(item_ids):

            top_k_indices = torch.topk(combined_similarity[i], k=min(top_k, len(self.set))).indices.tolist()

            item_to_insights[item_id] = top_k_indices
            
        logger.info(f"Created top-{top_k} insight map for {len(item_to_insights)} items")
        self.item_to_insights_map = item_to_insights
        
        self.encoder = None
        self.insight_text_embeddings = None
        self.insight_image_embeddings = None
        return item_to_insights
    
    def get_item_specific_insights(self, item_id, default_k=5, strategy='topk'):
        """Get the specific top-k insights for an item"""
        if strategy == 'topk':
            return self.get_cur_set_str(default_k)
        else:
            if item_id in self.item_to_insights_map:
                top_indices = self.item_to_insights_map[item_id]
                insights = [self.set[idx]['insight'] for idx in top_indices]
                return '\n'.join([f"{i}: {insight}" for i, insight in enumerate(insights)])
            else:

                return self.get_cur_set_str(default_k)

def process_single_item(row, model_id, model_params, img_dir, insight_set_manager, prompt_template, custom_prompt=None, strategy=None, dataset=None, num_frames=8, sample_strategy='uniform'):
    """
    Process a single item in a separate process

    Args:
        row: DataFrame row containing data information
        model_id: Model ID to use
        model_params: Model parameters
        img_dir: Directory containing video frames
        insight_set_manager: Manager for insights with item-specific mapping
        prompt_template: Template for the prompt
        custom_prompt: Custom prompt if provided
        dataset: Dataset name for loading metadata
        num_frames: Number of frames to sample
        sample_strategy: Strategy for sampling frames ('uniform' or 'sequential')

    Returns:
        Dictionary with processing result or None if failed
    """
    try:

        model = MLLMFactory(model_id, model_params)


        vid = str(row['vid'])
        

        try:
            frames = load_video_frames(img_dir, vid, max_frames=num_frames, sample_strategy=sample_strategy)
        except Exception as e:
            logger.warning(f"Failed to load video frames for {vid}: {e}")
            return None
        

        dataset_path = f"data/{dataset}"
        video_data = load_video_data(dataset_path, vid)
        text = video_data.get('text', '')
        label = row['label']


        insight_set_str = insight_set_manager.get_item_specific_insights(vid, strategy=strategy)


        prompt = custom_prompt.format(
            text=text,
            insight_set=insight_set_str
        )


        thought = model.chat_multi_img(prompt, frames)
        prediction = thought.split("Answer:")[-1].strip()
        note = 'success'


        if 'fake' in prediction.lower():
            prediction = 1
        elif 'real' in prediction.lower():
            prediction = 0
        elif '1' in prediction:
            prediction = 1
        elif '0' in prediction:
            prediction = 0
        else:
            prediction = 1
            note = prediction
            logger.warning(f"Prediction is not fake or real: {prediction}")

        return {
            "id": vid,
            "pred": prediction,
            "label": label,
            "thought": thought,
            "note": note
        }
    except Exception as e:
        logger.error(f"Error processing item {row.get('vid', 'unknown')}: {e}")
        return None

def process_single_item_with_model(row, model, img_dir, insight_set_manager, custom_prompt, strategy=None, dataset=None, num_frames=8, sample_strategy='uniform'):
    """
    Process a single item with a pre-initialized model (for single-process mode)

    Args:
        row: DataFrame row containing data information
        model: Pre-initialized model instance
        img_dir: Directory containing video frames
        insight_set_manager: Manager for insights with item-specific mapping
        custom_prompt: Custom prompt template
        strategy: Strategy for insight
        dataset: Dataset name for loading metadata
        num_frames: Number of frames to load
        sample_strategy: Frame sampling strategy

    Returns:
        Dictionary with processing result or None if failed
    """
    try:

        vid = str(row['vid'])
        

        try:
            frames = load_video_frames(img_dir, vid, max_frames=num_frames, sample_strategy=sample_strategy)
        except Exception as e:
            logger.warning(f"Failed to load video frames for {vid}: {e}")
            return None
        

        dataset_path = f"data/{dataset}"
        video_data = load_video_data(dataset_path, vid)
        text = video_data.get('text', '')
        label = row['label']


        insight_set_str = insight_set_manager.get_item_specific_insights(vid, strategy=strategy)


        prompt = custom_prompt.format(
            text=text,
            insight_set=insight_set_str
        )


        thought = model.chat_multi_img(prompt, frames)
        prediction = thought.split("Answer:")[-1].strip()
        note = 'success'


        if 'fake' in prediction.lower():
            prediction = 1
        elif 'real' in prediction.lower():
            prediction = 0
        elif '1' in prediction:
            prediction = 1
        elif '0' in prediction:
            prediction = 0
        else:
            prediction = 1
            note = prediction
            logger.warning(f"Prediction is not fake or real: {prediction}")

        return {
            "id": vid,
            "pred": prediction,
            "label": label,
            "thought": thought,
            "note": note
        }
    except Exception as e:
        logger.error(f"Error processing item {row.get('vid', 'unknown')}: {e}")
        return None

class InPredict_Runner():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.model_short_name = cfg.model_short_name
        self.dataset = cfg.dataset
        self.batch_size = cfg.batch_size
        self.cid = cid
        self.model_params = cfg.para if cfg.para else {}
        

        self.num_frames = OmegaConf.select(cfg, "num_frames", default=8)
        self.sample_strategy = OmegaConf.select(cfg, "sample_strategy", default="uniform")
        logger.info(f"Video frame config: num_frames={self.num_frames}, sample_strategy={self.sample_strategy}")
        
        self.log_dir = log_dir
        self.output_dir = self.log_dir
        self.img_dir = f"data/{self.dataset}/frames_16"
        self.insight_img_dir = f"data/{self.dataset}/insight_img" if os.path.exists(f"data/{self.dataset}/insight_img") else self.img_dir
        self.reference_name = cfg.reference_name
        self.data_split = cfg.get('data_split', 'train')
        


        reference_dir = 'reference/test' if self.data_split == 'test' else 'reference'
        reference_data = []
        with open(f"data/{self.dataset}/{reference_dir}/{self.reference_name}.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    reference_data.append(item)
        self.insight_df = pd.DataFrame(reference_data)
        
        self.data_file = f"data/{self.dataset}/label.jsonl"
        self.output_file = f"{self.output_dir}/inpred.jsonl"
        self.k = cfg.top_k
        
        self.prompt = OmegaConf.select(self.cfg, "prompt", default=prompt_template)
        logger.info(f"Using prompt: {self.prompt}")
        
        self.strategy = OmegaConf.select(self.cfg, "strategy", default="topk")

        self.insight_set_manager = InsightSetManager(size=30, strategy=self.strategy)
        self.insight_set_manager.init_from_df(self.insight_df)
        self.insight_str = self.insight_set_manager.get_cur_set_str(self.k)
        
        self.alpha = OmegaConf.select(self.cfg, "alpha", default=0.2)
        if OmegaConf.select(self.cfg, "extra_data"):
            self.extra_data = pd.read_json('data/FHM/model/PromptHate/clean_caption/mem.json')
        else:
            self.extra_data = None
        
    
    def log_result(self):
        result = self.get_result()
        logger.info(f"Accuracy: {result['acc']}, Macro F1: {result['f1']}, Macro Precision: {result['macro_prec']}, Macro Recall: {result['macro_rec']}")
        logger.info(f"Positive F1: {result['positive_f1']}, Negative F1: {result['negative_f1']}")
        
    def get_result(self):
        df = pd.read_json(self.output_file, lines=True)
        accuracy = accuracy_score(df['label'].astype(int), df['pred'].astype(int))
        f1 = f1_score(df['label'].astype(int), df['pred'].astype(int), average='macro')
        macro_prec = precision_score(df['label'].astype(int), df['pred'].astype(int), average='macro')
        macro_rec = recall_score(df['label'].astype(int), df['pred'].astype(int), average='macro')
        


        per_class_f1 = f1_score(df['label'].astype(int), df['pred'].astype(int), average=None)
        positive_f1 = per_class_f1[1] if len(per_class_f1) > 1 else 0.0
        negative_f1 = per_class_f1[0] if len(per_class_f1) > 0 else 0.0
        
        return {
            "acc": accuracy,
            "f1": f1,
            "macro_prec": macro_prec,
            "macro_rec": macro_rec,
            "positive_f1": positive_f1,
            "negative_f1": negative_f1
        }

    def run(self, num_processes=4):
        """
        Process a dataset using the specified MLLM model in parallel and save predictions
        
        Args:
            num_processes: Number of parallel processes to use
        """

        logger.info(f"Current template: {self.prompt}")
        os.makedirs(self.output_dir, exist_ok=True)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        output_file = self.output_file


        data_items = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data_items.append(item)
        input_df = pd.DataFrame(data_items)
        input_df['vid'] = input_df['vid'].astype(str)
        

        test_vids = set()

        split_file = f"data/{self.dataset}/vids/vid_time3_{self.data_split}.txt"
        with open(split_file, 'r') as f:
            for line in f:
                test_vids.add(line.strip())
        

        input_df = input_df[input_df['vid'].isin(test_vids)].reset_index(drop=True)
        logger.info(f"Loaded {len(input_df)} test samples")
        

        input_df = input_df.sample(frac=1, random_state=2025).reset_index(drop=True)
        

        processed_df = pd.DataFrame(columns=['id', 'pred', 'label', 'thought'])
        to_process_df = input_df
        

        if len(to_process_df) == 0:
            logger.info("No items to process.")
            return
        

        logger.info("Pre-computing embeddings and creating item-to-insights mapping with Jina CLIP")
        if self.strategy != 'topk':
            self.insight_set_manager.compute_item_embeddings_and_topk_map(
                input_df, 
                self.img_dir, 
                self.insight_img_dir, 
                top_k=self.k,
                alpha=self.alpha,
                dataset=self.dataset,
                num_frames=self.num_frames,
                sample_strategy=self.sample_strategy
            )
        

        items_to_process = list(to_process_df.iterrows())
        
        logger.info(f"Processing {len(items_to_process)} items using {num_processes} processes")


        process_func = partial(
            process_single_item,
            model_id=self.model_id,
            model_params=self.model_params,
            img_dir=self.img_dir,
            insight_set_manager=self.insight_set_manager,
            prompt_template=self.prompt,
            custom_prompt=self.prompt,
            strategy=self.strategy,
            dataset=self.dataset
        )

        results = []


        if num_processes <= 1:


            logger.info("Using single-process mode (suitable for CUDA models)")
            logger.info("Initializing model once for reuse across all items")
            model = MLLMFactory(self.model_id, self.model_params)

            for idx, row in tqdm(items_to_process, desc="Processing items"):
                result = process_single_item_with_model(
                    row,
                    model,
                    self.img_dir,
                    self.insight_set_manager,
                    self.prompt,
                    strategy=self.strategy,
                    dataset=self.dataset,
                    num_frames=self.num_frames,
                    sample_strategy=self.sample_strategy
                )
                if result:

                    processed_df = pd.concat([processed_df, pd.DataFrame([result])], ignore_index=True)
                    results.append(result)


                    if len(results) % 5 == 0:
                        processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

                        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
                        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
                        logger.info(f"Processed {len(results)} items. Current Accuracy: {accuracy:.4f}, F1: {f1:.4f}")


            processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        else:

            logger.info("Using multi-process mode (suitable for API models)")
            

            rows_only = [row for _, row in items_to_process]
            
            with multiprocessing.Pool(processes=min(num_processes, len(rows_only))) as pool:

                for result in tqdm(
                    pool.imap_unordered(process_func, rows_only),
                    total=len(rows_only),
                    desc="Processing items"
                ):
                    if result:

                        processed_df = pd.concat([processed_df, pd.DataFrame([result])], ignore_index=True)
                        results.append(result)


                        if len(results) % 5 == 0:
                            processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

                            accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
                            f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
                            logger.info(f"Processed {len(results)} items. Current Accuracy: {accuracy:.4f}, F1: {f1:.4f}")


                processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        

        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        

        pos_mask = processed_df['label'] == 1
        neg_mask = processed_df['label'] == 0


        positive_f1 = f1_score(
            processed_df['label'].astype(int),
            processed_df['pred'].astype(int),
            average='binary',
            pos_label=1
        ) if pos_mask.any() else 0.0


        negative_f1 = f1_score(
            processed_df['label'].astype(int),
            processed_df['pred'].astype(int),
            average='binary',
            pos_label=0
        ) if neg_mask.any() else 0.0


        logger.info(f"Positive F1: {positive_f1:.4f},  Negative F1: {negative_f1:.4f}")
        logger.info(f"Processing complete. Final Accuracy: {accuracy:.4f}, F1: {f1:.4f}")