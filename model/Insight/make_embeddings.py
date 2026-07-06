import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os
from pathlib import Path
import json






video_datasets = ['FakeTT', 'FVC']

model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=512, device='cuda')
batch_size = 2
max_image_size = 512








    





        






    





























    










for dataset in video_datasets:
    print(f"\n{'='*50}")
    print(f"Processing video dataset: {dataset}")
    print(f"{'='*50}")
    
    joint_fea_dict = {}
    image_fea_dict = {}
    text_fea_dict = {}


    labels_dict = {}
    label_file = Path(f'data/{dataset}/label.jsonl')
    if label_file.exists():
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())

                    vid = str(item['vid'])
                    labels_dict[vid] = item['label']
    

    available_vids = list(labels_dict.keys())
    print(f"Found {len(available_vids)} videos in label file for {dataset}")
    

    ocr_dict = {}
    ocr_file = Path(f'data/{dataset}/ocr.jsonl')
    if ocr_file.exists():
        with open(ocr_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    ocr_dict[str(item['vid'])] = item.get('text', '')
    

    transcript_dict = {}
    transcript_file = Path(f'data/{dataset}/transcript.jsonl')
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    transcript_dict[str(item['vid'])] = item.get('text', '')
    

    title_dict = {}
    title_file = Path(f'data/{dataset}/title.jsonl')
    if title_file.exists():
        with open(title_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    title_dict[str(item['vid'])] = item.get('text', '')
    

    frames_path = Path(f'data/{dataset}/frames_16')
    

    num_samples = len(available_vids)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc=f'Processing {dataset}'):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_vids = available_vids[start_idx:end_idx]
        
        batch_texts = []
        batch_images = []
        valid_vids = []
        
        for vid in batch_vids:

            vid_frames_dir = frames_path / vid
            if not vid_frames_dir.exists() or not vid_frames_dir.is_dir():
                print(f"Skipping {vid}: frames directory not found")
                continue
            

            text_parts = []
            if vid in title_dict:
                text_parts.append(title_dict[vid])
            if vid in ocr_dict:
                text_parts.append(ocr_dict[vid])
            if vid in transcript_dict:
                text_parts.append(transcript_dict[vid])
            text = ' '.join(text_parts).strip()

            

            frames = []
            for i in range(16):
                frame_path = vid_frames_dir / f'frame_{i:03d}.jpg'
                if frame_path.exists():
                    try:
                        img = Image.open(frame_path).convert("RGB")

                        img.thumbnail((max_image_size, max_image_size), Image.LANCZOS)
                        frames.append(img)
                    except Exception as e:
                        print(f"Error loading frame {i} for {vid}: {e}")
                        break
            

            if len(frames) == 16:
                try:

                    frame_width, frame_height = frames[0].size
                    

                    grid_width = frame_width * 4
                    grid_height = frame_height * 4
                    grid_image = Image.new('RGB', (grid_width, grid_height))
                    

                    for idx, frame in enumerate(frames):
                        row = idx // 4
                        col = idx % 4
                        x = col * frame_width
                        y = row * frame_height
                        grid_image.paste(frame, (x, y))
                    
                    batch_images.append(grid_image)
                    batch_texts.append(text)
                    valid_vids.append(vid)
                    

                    del frames
                except Exception as e:
                    print(f"Error creating grid for {vid}: {e}")
                    continue
            else:
                print(f"Skipping {vid}: only found {len(frames)} frames")
        
        if len(valid_vids) == 0:
            continue
        

        image_embeddings = model.encode(batch_images, normalize_embeddings=True, convert_to_tensor=True)
        text_embeddings = model.encode(batch_texts, normalize_embeddings=True, convert_to_tensor=True)
        
        for vid, img_emb, txt_emb in zip(valid_vids, image_embeddings, text_embeddings):
            join_embeddings = torch.cat([img_emb, txt_emb], dim=-1)
            joint_fea_dict[vid] = join_embeddings.cpu()
            image_fea_dict[vid] = img_emb.cpu()
            text_fea_dict[vid] = txt_emb.cpu()
        

        del batch_images, batch_texts, image_embeddings, text_embeddings
        torch.cuda.empty_cache()
    

    fea_path = Path(f'data/{dataset}/fea/')
    os.makedirs(fea_path, exist_ok=True)
    torch.save(joint_fea_dict, fea_path / 'joint_embed.pt')
    torch.save(image_fea_dict, fea_path / 'image_embed.pt')
    torch.save(text_fea_dict, fea_path / 'text_embed.pt')
    print(f"Saved embeddings for {len(joint_fea_dict)} samples from {dataset}")
