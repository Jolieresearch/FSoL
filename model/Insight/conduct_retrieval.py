
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def compute_batch_similarities(
    query_features_img: np.ndarray,
    query_features_txt: np.ndarray,
    base_features_img: np.ndarray,
    base_features_txt: np.ndarray,
    query_ids: List[int],
    base_ids: List[int],
    batch_size: int = 100,
) -> List[Dict]:
    """
    Compute similarities between query and base features in batches for both image and text
    """
    results = []

    for i in tqdm(range(0, len(query_features_img), batch_size)):
        batch_query_img = query_features_img[i : i + batch_size]
        batch_query_txt = query_features_txt[i : i + batch_size]
        batch_ids = query_ids[i : i + batch_size]


        similarities_img = 1 - cdist(batch_query_img, base_features_img, metric="cosine")
        similarities_txt = 1 - cdist(batch_query_txt, base_features_txt, metric="cosine")


        similarities = similarities_img + similarities_txt

        for idx, (qid, sims) in enumerate(zip(batch_ids, similarities)):
            matches = [
                {"vid": base_ids[i], "similarity": float(sims[i])}
                for i in range(len(sims))
                if qid != base_ids[i]
            ]

            matches.sort(key=lambda x: x["similarity"], reverse=True)

            results.append({"vid": qid, "matches": matches})

    return results










































def greedy_matching(
    similarities: List[Dict], data_df: pd.DataFrame, max_pairs: int = None
) -> List[Tuple[int, int, float]]:
    """
    Perform greedy matching based on similarities,
    ensuring that id1 (pred=0, authentic) is paired with id2 (pred=1, fake).
    This is critical for fake news detection: we want to pair authentic videos 
    with fake videos to generate contrastive insights.
    
    Returns:
        List of tuples (id1, id2, similarity) where id1 always has pred=0 and id2 has pred=1
    """
    all_pairs = []
    used_ids = set()
    data_df["pred"] = data_df["pred"].astype(int)
    

    vid_to_pred = dict(zip(data_df["vid"], data_df["pred"]))


    for item in similarities:
        query_vid = item["vid"]
        query_pred = vid_to_pred.get(query_vid)
        
        if query_pred is None:
            continue

        for match in item["matches"]:
            base_vid = match["vid"]
            base_pred = vid_to_pred.get(base_vid)
            
            if base_pred is None:
                continue




            if query_pred == 0 and base_pred == 1:

                all_pairs.append((query_vid, base_vid, match["similarity"]))
            elif query_pred == 1 and base_pred == 0:


                all_pairs.append((base_vid, query_vid, match["similarity"]))


    all_pairs.sort(key=lambda x: x[2], reverse=True)



    final_pairs = []
    for id1, id2, sim in all_pairs:
        if id1 not in used_ids and id2 not in used_ids:

            assert vid_to_pred[id1] == 0, f"id1 {id1} should have pred=0, got {vid_to_pred[id1]}"
            assert vid_to_pred[id2] == 1, f"id2 {id2} should have pred=1, got {vid_to_pred[id2]}"
            
            final_pairs.append((id1, id2, sim))
            used_ids.add(id1)
            used_ids.add(id2)

            if max_pairs and len(final_pairs) >= max_pairs:
                break

    return final_pairs


def random_pairing_wo_retrieval(
    data_df: pd.DataFrame, n_pairs: int = None, random_seed: int = 42
) -> List[Tuple[int, int, float]]:
    """
    Ablation study: w/o Pairing
    
    Remove the retrieval-based contrastive pairing mechanism.
    Randomly realign memes from two pseudo-categories to form new pairs.
    This breaks the semantic similarity constraint and tests the importance of proper pairing.
    
    Args:
        data_df: DataFrame with vid and pred columns
        n_pairs: Maximum number of pairs to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        List of tuples (id1, id2, similarity=0.0) where id1 has pred=0 and id2 has pred=1
    """
    np.random.seed(random_seed)
    data_df["pred"] = data_df["pred"].astype(int)
    

    authentic_vids = data_df[data_df["pred"] == 0]["vid"].tolist()
    fake_vids = data_df[data_df["pred"] == 1]["vid"].tolist()
    

    np.random.shuffle(authentic_vids)
    np.random.shuffle(fake_vids)
    

    max_possible_pairs = min(len(authentic_vids), len(fake_vids))
    if n_pairs:
        max_possible_pairs = min(max_possible_pairs, n_pairs)
    
    random_pairs = []
    for i in range(max_possible_pairs):

        random_pairs.append((authentic_vids[i], fake_vids[i], 0.0))
    
    print(f"Generated {len(random_pairs)} random pairs (w/o retrieval-based pairing)")
    return random_pairs



def main():

    parser = argparse.ArgumentParser(description='Conduct retrieval with configurable sampling mode and ablation studies')
    parser.add_argument('--sampling_mode', type=str, default='confidence', 
                        choices=['confidence', 'random'],
                        help="Sampling mode: 'confidence' (default) or 'random'")
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=['wo_pairing', 'wo_experience', 'wo_reference'],
                        help="Ablation mode: 'wo_pairing', 'wo_experience', or 'wo_reference' (optional)")
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['FakeTT', 'FakeSV', 'FVC'],
                        help='Datasets to process (default: FakeTT)')
    parser.add_argument('--model_name', type=str, default='test_Qwen3-VL-32B-Instruct',
                        help='Model name for predictions (default: Qwen3-VL-32B-Instruct)')
    parser.add_argument('--coverage_rates', type=float, nargs='+', 
                        default=[round(0.1 * i, 1) for i in range(1, 11)],
                        help='Coverage rates to process (default: 0.1 to 1.0)')
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'test'],
                        help='Data split to use: train or test (default: train)')
    
    args = parser.parse_args()
    

    datasets = args.datasets
    model_name = args.model_name
    coverage_rates = args.coverage_rates
    sampling_mode = args.sampling_mode
    ablation_mode = args.ablation_mode
    random_seed = args.random_seed
    data_split = args.data_split
    
    print("=" * 60)
    print("Configuration:")
    print(f"  Datasets: {datasets}")
    print(f"  Model: {model_name}")
    print(f"  Sampling mode: {sampling_mode}")
    if ablation_mode:
        print(f"  Ablation mode: {ablation_mode}")
    if sampling_mode == 'random' or ablation_mode == 'wo_pairing':
        print(f"  Random seed: {random_seed}")
    print(f"  Coverage rates: {coverage_rates}")
    print("=" * 60)


    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_path = Path("data") / dataset


        features_img = torch.load(dataset_path / "fea" / "image_embed.pt", weights_only=True)
        features_txt = torch.load(dataset_path / "fea" / "text_embed.pt", weights_only=True)


        label_dir = "label/test" if data_split == 'test' else "label"
        pred_file = dataset_path / label_dir / f"{model_name}.jsonl"
        pred_data = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    pred_data.append(item)
        pred_df = pd.DataFrame(pred_data)


        pred_df["vid"] = pred_df["vid"].astype(str)
        pred_df["pred"] = pred_df["pred"].astype(int)
        pred_df["label"] = pred_df["label"].astype(int)
        

        train_split_file = dataset_path / "vids" / f"vid_time3_{data_split}.txt"
        if train_split_file.exists():
            with open(train_split_file, 'r', encoding='utf-8') as f:
                train_vids = set([line.strip() for line in f.readlines() if line.strip()])

            pred_df = pred_df[pred_df["vid"].isin(train_vids)]
            print(f"Filtered to {len(pred_df)} training samples")
        else:
            print(f"Warning: Train split file not found for {dataset}, using all {len(pred_df)} samples")
        
        y_true = pred_df["label"]
        y_prob0 = pred_df["prob0"].astype(float)
        y_prob1 = pred_df["prob1"].astype(float)
        predicted_class = pred_df["pred"]


        pred_df["confidence"] = np.array(
            [y_prob1[i] if pred == 1 else y_prob0[i] for i, pred in enumerate(predicted_class)]
        )


        for coverage_rate in coverage_rates:
            print(f"\nProcessing with coverage rate: {coverage_rate:.1f}")
            print(f"Sampling mode: {sampling_mode}")


            n_samples = int(len(pred_df) * coverage_rate)
            

            if sampling_mode == 'confidence':

                high_conf_df = pred_df.sort_values("confidence", ascending=False).head(n_samples)
                print(f"Selected top {n_samples} samples by confidence")
            elif sampling_mode == 'random':

                np.random.seed(random_seed)
                random_indices = np.random.choice(len(pred_df), size=n_samples, replace=False)
                high_conf_df = pred_df.iloc[random_indices]
                print(f"Randomly selected {n_samples} samples (seed={random_seed})")
            else:
                raise ValueError(f"Invalid sampling_mode: {sampling_mode}. Must be 'confidence' or 'random'")

            
            high_conf_ids = high_conf_df["vid"].tolist()


            high_conf_y_true = high_conf_df["label"].astype(int)
            high_conf_y_pred = high_conf_df["pred"].astype(int)

            accuracy = accuracy_score(high_conf_y_true, high_conf_y_pred)
            f1 = f1_score(high_conf_y_true, high_conf_y_pred, average="macro")

            if sampling_mode == 'confidence':
                print(f"Metrics for top {coverage_rate:.1%} confidence samples:")
            else:
                print(f"Metrics for randomly selected {coverage_rate:.1%} samples:")
            print(
                f"Coverage: {coverage_rate:.1%} - Samples: {n_samples}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
            )
            print("----------------------------")


            filtered_high_conf_ids = high_conf_ids
            print(f"Number of selected samples: {len(filtered_high_conf_ids)}")


            if ablation_mode == 'wo_pairing':

                print("Ablation mode: w/o Pairing - Using random pairing without retrieval")
                matched_pairs = random_pairing_wo_retrieval(
                    high_conf_df, 
                    n_pairs=None,
                    random_seed=random_seed
                )
            else:


                high_conf_features_img = []
                high_conf_features_txt = []
                valid_ids = []

                for id_ in filtered_high_conf_ids:
                    if id_ in features_img and id_ in features_txt:
                        high_conf_features_img.append(features_img[id_].cpu().numpy())
                        high_conf_features_txt.append(features_txt[id_].cpu().numpy())
                        valid_ids.append(id_)

                print(f"Number of valid samples with features: {len(valid_ids)}")

                high_conf_features_img = np.array(high_conf_features_img)
                high_conf_features_txt = np.array(high_conf_features_txt)


                similarities = compute_batch_similarities(
                    high_conf_features_img,
                    high_conf_features_txt,
                    high_conf_features_img,
                    high_conf_features_txt,
                    valid_ids,
                    valid_ids,
                )


                matched_pairs = greedy_matching(similarities, pred_df)




            if ablation_mode == 'wo_pairing':
                output_path = dataset_path / "retrieve" / "test" / f"pairs_wo_pairing_{coverage_rate:.1f}.jsonl"
            elif sampling_mode == 'confidence':
                output_path = dataset_path / "retrieve" / "test" / f"pairs_coverage_{coverage_rate:.1f}.jsonl"
            else:
                output_path = dataset_path / "retrieve" / "test" / f"pairs_random_{coverage_rate:.1f}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for pair in matched_pairs:
                    result = {"id1": pair[0], "id2": pair[1], "similarity": pair[2]}
                    f.write(json.dumps(result) + "\n")

            print(f"Found {len(matched_pairs)} pairs")
            print(f"Results saved to {output_path}")
            print("----------------------------")


if __name__ == "__main__":
    main()
