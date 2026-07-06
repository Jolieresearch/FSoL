from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import json
import os
from typing import List
from loguru import logger

from .base_data import Base_Dataset


class FVC_Dataset(Base_Dataset):
    def __init__(self, split: str = None, **kwargs):
        super(FVC_Dataset, self).__init__()
        self.data_path = Path('data/FVC')
        self.frames_path = self.data_path / 'frames_16'
        self.data = self.get_data(split)
    
    def get_data(self, split: str = None) -> pd.DataFrame:
        """
        从jsonl文件加载FVC数据集
        """
        try:

            labels_dict = {}
            label_file = self.data_path / 'label.jsonl'
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())

                            labels_dict[str(item['vid'])] = item['label']
            

            title_dict = {}
            title_file = self.data_path / 'title.jsonl'
            if title_file.exists():
                with open(title_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            title_dict[str(item['vid'])] = item.get('text', '')
            

            ocr_dict = {}
            ocr_file = self.data_path / 'ocr.jsonl'
            if ocr_file.exists():
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            ocr_dict[str(item['vid'])] = item.get('text', '')
            

            transcript_dict = {}
            transcript_file = self.data_path / 'transcript.jsonl'
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            transcript_dict[str(item['vid'])] = item.get('text', '')
            

            available_vids = set()
            if self.frames_path.exists():
                available_vids = set(d.name for d in self.frames_path.iterdir() if d.is_dir())
            

            if split:
                split_file_map = {
                    'train': self.data_path / 'vids' / 'vid_time3_train.txt',
                    'valid': self.data_path / 'vids' / 'vid_time3_valid.txt', 
                    'test': self.data_path / 'vids' / 'vid_time3_test.txt',
                    'val': self.data_path / 'vids' / 'vid_time3_valid.txt'
                }
                
                if split not in split_file_map:
                    raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_file_map.keys())}")
                
                split_file = split_file_map[split]
                
                try:

                    with open(split_file, 'r', encoding='utf-8') as f:
                        split_ids = set([line.strip() for line in f.readlines() if line.strip()])
                    

                    target_vids = split_ids & available_vids
                    
                except FileNotFoundError:
                    logger.warning(f"Split file {split_file} not found. Using all available videos.")
                    target_vids = available_vids
                except Exception as e:
                    logger.error(f"Error reading split file {split_file}: {e}")
                    target_vids = available_vids
            else:

                target_vids = available_vids
            

            data_rows = []
            for vid in target_vids:

                text_parts = []
                title = title_dict.get(vid, '')
                ocr = ocr_dict.get(vid, '')
                transcript = transcript_dict.get(vid, '')
                
                if title:
                    text_parts.append(str(title))
                if ocr:
                    text_parts.append(str(ocr))
                if transcript:
                    text_parts.append(str(transcript))
                text = ' '.join(text_parts).strip()
                
                data_rows.append({
                    'vid': vid,
                    'title': title,
                    'ocr': ocr,
                    'transcript': transcript,
                    'text': text,
                    'label': labels_dict.get(vid, 0)
                })
            
            data = pd.DataFrame(data_rows)
            split_info = f" for {split} split" if split else ""
            logger.info(f"Loaded FVC dataset{split_info} with {len(data)} samples")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading FVC dataset: {e}")
            return pd.DataFrame()
    
    def get_full_data(self) -> pd.DataFrame:
        return self.get_data(split=None)

    def get_frames(self, vid: str) -> List[Image.Image]:
        """获取视频的16帧图像"""
        frames_dir = self.frames_path / vid
        frames = []
        

        for i in range(16):
            frame_path = frames_dir / f"frame_{i:03d}.jpg"
            if frame_path.exists():
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            else:

                if frames:
                    frames.append(frames[-1])
                else:

                    blank_frame = Image.new('RGB', (224, 224), color='white')
                    frames.append(blank_frame)
        
        return frames

    def get_img(self, vid: str) -> Image.Image:
        """获取第一帧作为单张图像（向后兼容）"""
        frames = self.get_frames(vid)
        return frames[0] if frames else Image.new('RGB', (224, 224), color='white')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        frames = self.get_frames(vid)
        img = self.get_img(vid)
        label = item['label']

        text_parts = []
        if item.get('title'):
            text_parts.append(str(item['title']))
        if item.get('ocr'):
            text_parts.append(str(item['ocr']))
        if item.get('transcript'):
            text_parts.append(str(item['transcript']))
        text = ' '.join(text_parts).strip()
        
        return {
            'id': vid,
            'vid': vid,
            'img': img,
            'frames': frames,
            'label': label,
            'text': text
        }


class FVC_Collator:
    def __init__(self):
        pass

    def __call__(self, batch):
        vid = [item['id'] for item in batch]
        img = [item['img'] for item in batch]
        frames = [item['frames'] for item in batch]
        label = [item['label'] for item in batch]
        text = [item['text'] for item in batch]
        return {
            'id': vid,
            'img': img,
            'frames': frames,
            'label': label,
            'text': text
        }