from ..utils.mllm_factory import MLLMFactory
from ..utils.video_utils import load_video_frames, load_video_data
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

class Insight_Runner:
    def __init__(self, cfg: DictConfig, **kwargs):
        """
        Initialize Insight Runner for fake video datasets
        
        Args:
            model_id: ID of the model to use
        """
        self.model_id = cfg.model_id
        self.model_params = cfg.para if cfg.para else {}
        self.model = MLLMFactory(self.model_id, self.model_params)
        self.dataset = cfg.dataset
        self.num_frames = cfg.get('num_frames', 8)
        self.sample_strategy = cfg.get('sample_strategy', 'uniform')
        logger.info(f"Using {self.num_frames} frames with {self.sample_strategy} sampling strategy")
        self.frames_path = f'data/{self.dataset}/frames_16'
        self.dataset_path = f'data/{self.dataset}'
        self.result = None
        self.pairs_name = cfg.pairs_name
        self.insight_namae = cfg.insight_name
        self.data_split = cfg.get('data_split', 'train')
        


        pair_data = []
        with open(f'data/{self.dataset}/retrieve/test/{self.pairs_name}.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    pair_data.append(item)
        self.pair_df = pd.DataFrame(pair_data)
        self.pair_df['id1'] = self.pair_df['id1'].astype(str)
        self.pair_df['id2'] = self.pair_df['id2'].astype(str)
        

        self.video_metadata = {}
        for vid_field in ['id1', 'id2']:
            unique_vids = self.pair_df[vid_field].unique()
            for vid in unique_vids:
                if vid not in self.video_metadata:
                    self.video_metadata[vid] = load_video_data(self.dataset_path, vid)
        



        self.pair_df = self.pair_df.sort_values(by='similarity', ascending=False)

        insight_dir = 'insight/test' if self.data_split == 'test' else 'insight'
        self.save_path = f'data/{cfg.dataset}/{insight_dir}/{self.insight_namae}.jsonl'
        

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        

        if os.path.exists(self.save_path):
            try:
                existing_data = []
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            existing_data.append(item)
                self.save_df = pd.DataFrame(existing_data).astype({'id1': str, 'id2': str})
                logger.info(f"Loaded {len(self.save_df)} existing results from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}. Starting fresh.")
                self.save_df = pd.DataFrame(columns=['id1', 'id2', 'insight']).astype({'id1': str, 'id2': str})
        else:
            logger.info(f"No existing results found. Starting fresh.")
            self.save_df = pd.DataFrame(columns=['id1', 'id2', 'insight']).astype({'id1': str, 'id2': str})
        
    def preprocess(self):
        pass
    
    def run(self, num_processes=1):
        """
        Run inference on multiple video pairs using multiprocessing
        
        Args:
            num_processes: Number of parallel processes to use (default=1 for video processing)
        
        Returns:
            None: Results are saved to the output file
        """

        pairs_to_process = []
        for _, row in self.pair_df.iterrows():

            if len(self.save_df) > 0 and ((self.save_df['id1'] == row['id1']) & (self.save_df['id2'] == row['id2'])).any():
                continue
            pairs_to_process.append((row['id1'], row['id2']))
        

        if not pairs_to_process:
            logger.info("No pairs to process.")
            return
        
        logger.info(f"Processing {len(pairs_to_process)} pairs {'sequentially' if num_processes == 1 else f'using {num_processes} processes'}")
        
        if num_processes == 1:

            results = []
            for pair_info in tqdm(pairs_to_process, desc="Processing pairs"):
                result = self.process_pair_single(pair_info)
                if result is not None:
                    results.append(result)
                

                if len(results) % 5 == 0 and len(results) > 0:
                    new_results_df = pd.DataFrame(results)
                    combined_df = pd.concat([self.save_df, new_results_df], ignore_index=True).astype({'id1': str, 'id2': str})
                    combined_df.to_json(self.save_path, lines=True, index=False, orient='records')
                    logger.info(f"Saved {len(results)} results so far")
                    self.save_df = combined_df
                    results = []
            

            if results:
                new_results_df = pd.DataFrame(results)
                self.save_df = pd.concat([self.save_df, new_results_df], ignore_index=True).astype({'id1': str, 'id2': str})
                self.save_df.to_json(self.save_path, lines=True, index=False, orient='records')
        else:

            process_func = partial(
                process_pair,
                model_id=self.model_id,
                model_params=self.model_params,
                frames_path=self.frames_path,
                dataset_path=self.dataset_path,
                video_metadata=self.video_metadata
            )
            

            with multiprocessing.Pool(processes=min(num_processes, len(pairs_to_process))) as pool:

                results = []
                for result in tqdm(
                    pool.imap_unordered(process_func, pairs_to_process),
                    total=len(pairs_to_process),
                    desc="Processing pairs"
                ):
                    if result is not None:
                        results.append(result)
                    

                    if len(results) % 5 == 0 and len(results) > 0:
                        new_results_df = pd.DataFrame(results)
                        combined_df = pd.concat([self.save_df, new_results_df], ignore_index=True).astype({'id1': str, 'id2': str})
                        combined_df.to_json(self.save_path, lines=True, index=False, orient='records')
                        logger.info(f"Saved {len(results)} results so far")
                        self.save_df = combined_df
                        results = []
                

                if results:
                    new_results_df = pd.DataFrame(results)
                    self.save_df = pd.concat([self.save_df, new_results_df], ignore_index=True).astype({'id1': str, 'id2': str})
                    self.save_df.to_json(self.save_path, lines=True, index=False, orient='records')
        
        logger.info("All pairs processed successfully")
    
    def process_pair_single(self, pair_info):
        """
        Process a single pair using the pre-loaded model instance
        
        Args:
            pair_info: Tuple containing (id1, id2)
            
        Returns:
            Dictionary with insight result
        """
        id1, id2 = pair_info
        

        try:
            vid1_frames = load_video_frames(self.frames_path, id1, max_frames=self.num_frames, sample_strategy=self.sample_strategy)
            vid2_frames = load_video_frames(self.frames_path, id2, max_frames=self.num_frames, sample_strategy=self.sample_strategy)
        except Exception as e:
            logger.error(f"Error loading video frames for pair ({id1}, {id2}): {e}")
            return None
        

        text1 = self.video_metadata.get(id1, {}).get('text', '')
        text2 = self.video_metadata.get(id2, {}).get('text', '')
        

        prompt = f"Video1 Text: {text1}\nVideo2 Text: {text2}\n"
        prompt += (
            "You are given two videos, each represented by individual frames plus textual elements.\n"
            "Video1 is authentic/real, while Video2 contains fake or misleading information.\n\n"
            "Carefully analyze BOTH videos by considering:\n"
            "- Visual content: objects, people, events, and their relationships\n"
            "- Text-visual alignment: does the text match what is shown?\n"
            "- Semantic coherence: does the narrative make logical sense?\n"
            "- Signs of manipulation: visual artifacts, implausible events, contradictions\n\n"
            "Note: Do NOT use frame repetition or sampling patterns as evidence, as consecutive video frames naturally contain similar content.\n\n"
            "Then complete the following tasks:\n"
            "1. Provide a concise description of the content of each video in one or two sentences.\n"
            "2. Contrast the two videos to explain WHY Video2 is fake/misleading while Video1 is authentic. "
            "Focus on: mismatched text-visual cues, implausible events, visual artifacts, logical contradictions, or narrative incoherence.\n"
            "3. Provide ONE concrete and generalizable detection rule for identifying this type of fake video manipulation.\n"
        )

        


        all_frames = vid1_frames + vid2_frames
        insight = self.model.chat_multi_img(prompt, all_frames)
        
        return {
            'id1': id1,
            'id2': id2,
            'insight': insight
        }
        
    def log_result(self):
        pass


def process_pair(pair_info, model_id, model_params, frames_path, dataset_path, video_metadata, num_frames=8, sample_strategy='uniform'):
    """
    Process a single pair of videos in a separate process
    
    Args:
        pair_info: Tuple containing (id1, id2)
        model_id: Model ID to use
        model_params: Model parameters
        frames_path: Path to video frames directory
        dataset_path: Path to dataset directory
        video_metadata: Dictionary containing preloaded video metadata
        num_frames: Number of frames to sample
        sample_strategy: Strategy for sampling frames ('uniform' or 'sequential')
        
    Returns:
        Dictionary with insight result
    """

    model = MLLMFactory(model_id, model_params)
    
    id1, id2 = pair_info
    

    try:
        vid1_frames = load_video_frames(frames_path, id1, max_frames=num_frames, sample_strategy=sample_strategy)
        vid2_frames = load_video_frames(frames_path, id2, max_frames=num_frames, sample_strategy=sample_strategy)
    except Exception as e:
        logger.error(f"Error loading video frames for pair ({id1}, {id2}): {e}")
        return None
    

    text1 = video_metadata.get(id1, {}).get('text', '')
    text2 = video_metadata.get(id2, {}).get('text', '')
    






    prompt = f"Video1 Text: {text1}\nVideo2 Text: {text2}\n"
    prompt += (
        "You are given two videos, each represented by individual frames plus textual elements.\n"
        "Video1 is authentic/real, while Video2 contains fake or misleading information.\n\n"
        "Carefully analyze BOTH videos by considering:\n"
        "- Visual content: objects, people, events, and their relationships\n"
        "- Text-visual alignment: does the text match what is shown?\n"
        "- Semantic coherence: does the narrative make logical sense?\n"
        "- Signs of manipulation: visual artifacts, implausible events, contradictions\n\n"
        "Note: Do NOT use frame repetition or sampling patterns as evidence, as consecutive video frames naturally contain similar content.\n\n"
        "Then complete the following tasks:\n"
        "1. Provide a concise description of the content of each video in one or two sentences.\n"
        "2. Contrast the two videos to explain WHY Video2 is fake/misleading while Video1 is authentic. "
        "Focus on: mismatched text-visual cues, implausible events, visual artifacts, logical contradictions, or narrative incoherence.\n"
        "3. Provide ONE concrete and generalizable detection rule for identifying this type of fake video manipulation.\n"
    )

    


    all_frames = vid1_frames + vid2_frames
    insight = model.chat_multi_img(prompt, all_frames)
    
    return {
        'id1': id1,
        'id2': id2,
        'insight': insight
    }