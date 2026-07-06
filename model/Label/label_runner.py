from ..utils.mllm_factory import MLLMFactory
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from ..utils.data_factory import DataDfFactory
import torch
import gc


class Label_Runner():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        self.cid = cid
        self.log_dir = log_dir
        self.output_dir = self.log_dir

        

        self.label_name = cfg.label_name
        self.output_file = Path('data') / self.cfg.dataset / 'label' / f'{self.label_name}.jsonl'
        
        if self.cfg.prompt:
            logger.info(f"Using custom prompt: {self.cfg.prompt}")
        self.para = self.cfg.para if self.cfg.para else {}
        
        self.model = MLLMFactory(self.cfg.model_id, self.para)
        self.datadf = DataDfFactory(self.cfg.dataset, **self.cfg.data)
        self.prompt = self.cfg.prompt if self.cfg.prompt else ""
        

        memory_config = getattr(cfg, 'memory_config', {})
        self.max_image_size = memory_config.get('max_image_size', getattr(cfg, 'max_image_size', 384))
        self.max_frames = memory_config.get('max_frames', getattr(cfg, 'max_frames', 8))
        self.memory_cleanup_interval = getattr(cfg, 'memory_cleanup_interval', 10)
        self.batch_size = getattr(cfg, 'batch_size', 1)
        
        logger.info(f"Memory optimization settings: max_image_size={self.max_image_size}, max_frames={self.max_frames}")
        
    def resize_image_for_memory(self, image: Image.Image, max_size: int = None) -> Image.Image:
        """调整图像大小以降低显存使用"""
        if max_size is None:
            max_size = self.max_image_size
            
        width, height = image.size
        

        if width <= max_size and height <= max_size:
            return image
            

        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
            
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def sample_frames(self, frames: list, max_frames: int = None) -> list:
        """采样帧以降低显存使用"""
        if max_frames is None:
            max_frames = self.max_frames
            
        if len(frames) <= max_frames:
            return frames
        

        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def _parse_prediction_text(self, prediction_text: str) -> int:
        """鲁棒地从模型输出文本中解析0/1预测"""
        prediction_text_clean = prediction_text.strip()
        prediction_text_lower = prediction_text_clean.lower()
        
        logger.debug(f"Parsing prediction from: {prediction_text_clean[:100]}...")
        

        import re
        first_digit = re.search(r'\b[01]\b', prediction_text_clean)
        if first_digit:
            result = int(first_digit.group())
            logger.debug(f"Found digit {result} via regex")
            return result
        


        fake_keywords = ['fake', 'fabricat', 'manipulat', 'misleading', 'false', 'deceptive', 'synthetic']
        real_keywords = ['real', 'authentic', 'genuine', 'trustworthy', 'legitimate', 'true']
        
        has_fake = any(kw in prediction_text_lower for kw in fake_keywords)
        has_real = any(kw in prediction_text_lower for kw in real_keywords)
        
        if has_fake and not has_real:
            logger.debug("Detected 'fake' keywords")
            return 1
        elif has_real and not has_fake:
            logger.debug("Detected 'real' keywords")
            return 0
        

        if len(prediction_text_clean) <= 3:
            if '1' in prediction_text_clean:
                return 1
            elif '0' in prediction_text_clean:
                return 0
        

        logger.warning(f"Cannot confidently parse prediction from: {prediction_text_clean[:200]}")
        return 1
    
    def log_result(self):
        if not os.path.exists(self.output_file):
            logger.warning(f"Output file {self.output_file} does not exist. Skipping log_result.")
            return
        
        try:
            processed_df = pd.read_json(self.output_file, lines=True)
            if len(processed_df) == 0:
                logger.warning("Output file is empty. Skipping log_result.")
                return
        except Exception as e:
            logger.error(f"Error reading output file: {e}")
            return
            
        y_true = processed_df['label'].astype(int)
        y_prob0 = processed_df['prob0'].astype(float)
        y_prob1 = processed_df['prob1'].astype(float)



        predicted_class = (y_prob1 >= 0.5).astype(int)
        confidence_scores = np.array([y_prob1[i] if pred == 1 else y_prob0[i] for i, pred in enumerate(predicted_class)])
        

        thresholds = np.arange(0.999, 1.0, 0.00001)
        
        total_samples = len(y_true)
        
        for threshold in thresholds:

            confident_idx = confidence_scores >= threshold
            

            y_true_confident = y_true[confident_idx]
            y_pred_confident = (y_prob1[confident_idx] >= 0.5).astype(int)
            

            if len(y_true_confident) > 0:
                acc = accuracy_score(y_true_confident, y_pred_confident)
                f1 = f1_score(y_true_confident, y_pred_confident, average='macro')
                coverage = len(y_true_confident) / total_samples
                
                logger.info(f"Confidence threshold {threshold:.5f} - Total samples: {total_samples}, Samples above threshold: {len(y_true_confident)}, Coverage: {coverage:.2%}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
            else:
                logger.info(f"No samples above confidence threshold {threshold:.1f} out of {total_samples} total samples")
                

        sorted_indices = np.argsort(confidence_scores)[::-1]
        sorted_confidences = confidence_scores[sorted_indices]
        logger.info(f"Top 10 confidences: {sorted_confidences[:10]}")
        sorted_y_true = y_true.iloc[sorted_indices]
        sorted_y_pred = predicted_class[sorted_indices]
        
        total_samples = len(y_true)
        

        coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        
        logger.info("Performance at different coverage levels:")
        logger.info("--------------------------------")
        

        logger.info(f"Verifying confidence scores are properly sorted (descending): {np.all(np.diff(sorted_confidences) <= 0)}")
        
        for coverage in coverage_levels:

            n_samples = int(total_samples * coverage)
            
            if n_samples == 0:
                continue
                

            threshold = sorted_confidences[n_samples-1] if n_samples < len(sorted_confidences) else 0
            

            if coverage == 0.1:
                logger.info(f"Confidence scores at positions 0, n_samples/2, n_samples-1: "
                           f"{sorted_confidences[0]:.5f}, {sorted_confidences[n_samples//2]:.5f}, {sorted_confidences[n_samples-1]:.5f}")
            

            current_y_true = sorted_y_true.iloc[:n_samples]
            current_y_pred = sorted_y_pred[:n_samples]
            
            acc = accuracy_score(current_y_true, current_y_pred)
            f1 = f1_score(current_y_true, current_y_pred, average='macro')
            
            logger.info(f"Coverage: {coverage:.1%} - Threshold: {threshold:.5f}, Samples: {n_samples}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
             
    def get_result(self):
        if not os.path.exists(self.output_file):
            logger.warning(f"Output file {self.output_file} does not exist.")
            return {"acc": 0.0, "f1": 0.0, "macro_prec": 0.0, "macro_rec": 0.0}
        
        try:
            processed_df = pd.read_json(self.output_file, lines=True)
            if len(processed_df) == 0:
                logger.warning("Output file is empty.")
                return {"acc": 0.0, "f1": 0.0, "macro_prec": 0.0, "macro_rec": 0.0}
        except Exception as e:
            logger.error(f"Error reading output file: {e}")
            return {"acc": 0.0, "f1": 0.0, "macro_prec": 0.0, "macro_rec": 0.0}
            
        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        macro_prec = precision_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        macro_rec = recall_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        return {
            "acc": accuracy,
            "f1": f1,
            "macro_prec": macro_prec,
            "macro_rec": macro_rec
        }

    def run(self, num_processes):
        """
        Process a dataset using the specified MLLM model and save predictions
        """
        output_file = self.output_file


        input_df = self.datadf
        input_df = input_df.sample(frac=1).reset_index(drop=True)


        all_records = []

        accuracy = 0.0
        f1 = 0.0
        pbar = tqdm(total=len(input_df), desc="Processing", position=0, leave=True)

        for idx, row in input_df.iterrows():
            pbar.set_description(f"ID: {row['vid']}, ACC: {accuracy:.2f}, F1: {f1:.2f}")

            text = row.get("text", "")

            label = row["label"]


            if self.cfg.dataset in ['FakeTT', 'FakeSV', 'FVC']:
                frames_dir = f"data/{self.cfg.dataset}/frames_16/{row['vid']}"
                if not os.path.exists(frames_dir):
                    logger.debug(f"Frames directory not found: {frames_dir}")
                    pbar.update(1)
                    continue

                frames = []
                frame_indices = np.linspace(0, 15, self.max_frames, dtype=int)

                for i in frame_indices:
                    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.jpg")
                    if os.path.exists(frame_path):
                        frame = Image.open(frame_path).convert("RGB")
                        frame = self.resize_image_for_memory(frame)
                        frames.append(frame)
                    else:
                        if frames:
                            frames.append(frames[-1])
                        else:
                            blank_frame = Image.new('RGB', (self.max_image_size, self.max_image_size), color='white')
                            frames.append(blank_frame)

                if len(frames) == 0:
                    pbar.update(1)
                    continue


                prompt = (
                    f"Given this sequence of video frames with the Text: '{text}', classify whether this video is FAKE or REAL.\n\n"
                    "INSTRUCTIONS:\n"
                    "- Output ONLY a single digit\n"
                    "- Output '1' if FAKE\n"
                    "- Output '0' if REAL\n"
                    "- Do NOT explain, do NOT add any other text\n\n"
                    "Answer (0 or 1):"
                )


                if self.prompt:
                    prompt = f"Given this sequence of video frames with the Text: '{text}'\n{self.prompt}"


                if hasattr(self.model, 'chat_multi_img_label'):
                    try:
                        prediction_text, probs = self.model.chat_multi_img_label(prompt, frames)

                        if isinstance(prediction_text, str):
                            prediction = self._parse_prediction_text(prediction_text)
                        else:
                            prediction = prediction_text
                    except Exception as e:
                        logger.error(f"chat_multi_img_label failed: {e}")

                        prediction_text = self.model.chat_multi_img(prompt, frames)
                        prediction = self._parse_prediction_text(prediction_text)
                        probs = [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
                elif hasattr(self.model, 'chat_multi_img'):
                    prediction_text = self.model.chat_multi_img(prompt, frames)
                    prediction = self._parse_prediction_text(prediction_text)
                    probs = [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
                else:
                    prediction, probs = self.model.chat_label(prompt, frames[0])

                del frames


            else:
                img_path = os.path.join(self.img_dir, row["img"])
                if not os.path.exists(img_path):
                    pbar.update(1)
                    continue

                image = Image.open(img_path).convert("RGB")
                image = self.resize_image_for_memory(image)

                prompt = (
                    f"Given the meme, with the Text: '{text}' embedded in the image, your task is to determine whether this "
                    "meme is harmful or not, in order to maintain the harmlessness and integrity of information on the Internet. "
                    "Please leverage your extensive knowledge to deeply analyze and understand this meme, and give your final judgment. "
                    "You must only return '1' for harmful, or '0' for harmless."
                )
                if self.prompt:
                    prompt = f"Given the meme, with the Text: '{text}' embedded in the image\n{self.prompt}"

                prediction, probs = self.model.chat_label(prompt, image)
                del image


            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



            if not isinstance(prediction, int):
                try:
                    prediction = int(prediction)
                except (ValueError, TypeError):
                    logger.error(f"Failed to convert prediction to int: {prediction}")

                    prediction = 1
            
            output_dict = {
                "vid": row["vid"],
                "pred": prediction,
                "label": label,
                "prob0": probs[0],
                "prob1": probs[1],
            }
            all_records.append(output_dict)


            tmp_df = pd.DataFrame(all_records)
            accuracy = accuracy_score(tmp_df['label'].astype(int), tmp_df['pred'].astype(int))
            f1 = f1_score(tmp_df['label'].astype(int), tmp_df['pred'].astype(int), average='macro')


            tmp_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

            pbar.update(1)

        pbar.close()


        final_df = pd.DataFrame(all_records)
        if len(final_df) > 0:
            final_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
            

            accuracy = accuracy_score(final_df['label'].astype(int), final_df['pred'].astype(int))
            f1 = f1_score(final_df['label'].astype(int), final_df['pred'].astype(int), average='macro')
            macro_prec = precision_score(final_df['label'].astype(int), final_df['pred'].astype(int), average='macro')
            macro_rec = recall_score(final_df['label'].astype(int), final_df['pred'].astype(int), average='macro')
            logger.info(f"Final Results - Accuracy: {accuracy:.4f}, Macro F1: {f1:.4f}, Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}")
        else:
            logger.warning("No samples were processed successfully!")