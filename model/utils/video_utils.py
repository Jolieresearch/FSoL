"""
Video processing utilities for fake video datasets
"""
from PIL import Image
from pathlib import Path
from typing import List
import os


def load_video_frames(frames_dir: str, vid: str, max_frames: int = 16, sample_strategy: str = 'uniform', max_size: int = None) -> List[Image.Image]:
    """
    Load video frames as a list of PIL Images
    
    Args:
        frames_dir: Base directory containing video frames (e.g., 'data/FakeTT/frames_16')
        vid: Video ID
        max_frames: Maximum number of frames to load (default: 16)
        sample_strategy: Sampling strategy
            - 'sequential': Take first max_frames frames (0,1,2,...,max_frames-1)
            - 'uniform': Sample frames uniformly with step=2 (0,2,4,6,...,2*max_frames-2)
            - 'all': Take all available frames up to max_frames
        max_size: Maximum image size (width/height). If specified, resize images while maintaining aspect ratio.
        
    Returns:
        List of PIL Images containing the video frames
    """
    vid_frames_dir = Path(frames_dir) / str(vid)
    
    if not vid_frames_dir.exists():
        raise FileNotFoundError(f"Video frames directory not found: {vid_frames_dir}")
    

    all_frame_files = sorted(vid_frames_dir.glob("*.jpg"))
    

    if sample_strategy == 'uniform':

        required_frames = max_frames * 2
        if len(all_frame_files) < required_frames:
            raise ValueError(f"Expected at least {required_frames} frames for uniform sampling of {max_frames} frames, found {len(all_frame_files)} for vid {vid}")
        frame_files = [all_frame_files[i * 2] for i in range(max_frames)]
    elif sample_strategy == 'sequential':

        if len(all_frame_files) < max_frames:
            raise ValueError(f"Expected at least {max_frames} frames, found {len(all_frame_files)} for vid {vid}")
        frame_files = all_frame_files[:max_frames]
    else:
        frame_files = all_frame_files[:max_frames]
    

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        

        if max_size is not None:
            width, height = img.size
            if width > max_size or height > max_size:

                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                img = img.resize((new_width, new_height), Image.LANCZOS)
        
        frames.append(img)
    
    return frames


def load_video_frames_as_grid(frames_dir: str, vid: str, grid_size: tuple = (4, 4), 
                               frame_size: tuple = (224, 224)) -> Image.Image:
    """
    Load 16 frames from a video directory and create a 4x4 grid image
    
    Args:
        frames_dir: Base directory containing video frames (e.g., 'data/FakeTT/frames_16')
        vid: Video ID
        grid_size: Grid dimensions (rows, cols)
        frame_size: Size to resize each frame to
        
    Returns:
        PIL Image containing the 4x4 grid of frames
    """
    vid_frames_dir = Path(frames_dir) / str(vid)
    
    if not vid_frames_dir.exists():
        raise FileNotFoundError(f"Video frames directory not found: {vid_frames_dir}")
    

    frame_files = sorted(vid_frames_dir.glob("*.jpg"))
    
    if len(frame_files) < 16:
        raise ValueError(f"Expected 16 frames, found {len(frame_files)} for vid {vid}")
    

    frame_files = frame_files[:16]
    

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        img = img.resize(frame_size, Image.Resampling.LANCZOS)
        frames.append(img)
    

    rows, cols = grid_size
    grid_width = frame_size[0] * cols
    grid_height = frame_size[1] * rows
    grid_img = Image.new('RGB', (grid_width, grid_height))
    
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_size[0]
        y = row * frame_size[1]
        grid_img.paste(frame, (x, y))
    
    return grid_img


def load_video_data(dataset_path: str, vid: str) -> dict:
    """
    Load video metadata from jsonl files
    
    Args:
        dataset_path: Path to dataset directory
        vid: Video ID
        
    Returns:
        Dictionary containing video metadata (label, title, ocr, transcript, text)
    """
    import json
    
    dataset_path = Path(dataset_path)
    vid = str(vid)
    
    result = {'vid': vid}
    

    label_file = dataset_path / 'label.jsonl'
    if label_file.exists():
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if str(item['vid']) == vid:
                        result['label'] = item['label']
                        break
    

    title_file = dataset_path / 'title.jsonl'
    if title_file.exists():
        with open(title_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if str(item['vid']) == vid:
                        result['title'] = item.get('text', '')
                        break
    

    ocr_file = dataset_path / 'ocr.jsonl'
    if ocr_file.exists():
        with open(ocr_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if str(item['vid']) == vid:
                        result['ocr'] = item.get('text', '')
                        break
    

    transcript_file = dataset_path / 'transcript.jsonl'
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if str(item['vid']) == vid:
                        result['transcript'] = item.get('text', '')
                        break
    

    text_parts = []
    if 'title' in result and result['title']:
        text_parts.append(f"Title: {result['title']}")
    if 'ocr' in result and result['ocr']:
        text_parts.append(f"OCR: {result['ocr']}")
    if 'transcript' in result and result['transcript']:
        text_parts.append(f"Transcript: {result['transcript']}")
    
    result['text'] = ' | '.join(text_parts) if text_parts else ''
    
    return result
