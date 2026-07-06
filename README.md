# FSoL

This repo provides an official implementation of **FSoL** as described in the paper: *Enabling Zero-Shot Fake News Video Detection with Training-Free Self-Optimization of LVLMs*.


## Source Code Structure

```sh
├── config                 # Hydra configs for every stage / dataset / backbone
│   ├── label_*.yaml       # Stage 1: confidence-aware pseudo-annotation
│   ├── insight_*.yaml     # Stage 3: discrepancy mining
│   ├── manage_*.yaml      # Stage 3: heuristic compilation
│   └── inpredict_*.yaml   # Stage 4: cue-guided inference
├── main.py                # unified entry point (dispatches by cfg.task)
├── model
│   ├── Label              # Stage 1: pseudo-annotation via LVLM confidence
│   ├── Insight            # Stage 2 & 3: embedding, retrieval, discrepancy mining
│   ├── Manage             # Stage 3: heuristic set compilation
│   ├── InPredict          # Stage 4: cue-guided inference
│   └── utils              # utils for datasets and LVLM invocation
├── utils                  # utils for the main code
├── script
│   └── run_fsol_qwen.sh   # end-to-end pipeline (FakeTT / FakeSV / FVC)
├── requirements.txt
└── .env                   # API key / URL and number of processes
```

## Dataset

Due to copyright restrictions, the raw datasets are not included in this repository. You can obtain them from their respective original project sites.

+ [FakeSV](https://github.com/ICTMCG/FakeSV)
+ [FakeTT](https://github.com/ICTMCG/FakingRecipe)
+ [FVC](https://mklab.iti.gr/results/fake-video-corpus/)

## Usage

### Requirement

To set up the environment, run the following commands:

```sh
conda create --name FSoL python=3.12
conda activate FSoL
pip install -r requirements.txt
```

### Prepare dataset

1. Download the datasets and store them under a `data` directory following the structure above, saving the video frames of each dataset to the corresponding path.
2. For each dataset, provide a `data.jsonl` file, with each line containing `id`, `text`, `label`, `video` (frame path), and `split`. An example is provided below:
```
{"id": 12345, "video": "12345", "label": 1, "text": "the accompanying title / OCR / transcript of the video", "split": "train"}
```

### Configure the API / model

FSoL supports both locally deployed LVLMs and API-based inference. Fill in your own credentials in `.env`:

```sh
API_KEY=<your_api_key>
API_URL=<your_api_url>
NUM_PROCESSES=1
```

> **Note:** The Confidence-Aware Pseudo-Annotation stage (`label`) must be performed by a **locally deployed** model, since it requires direct access to the *Fake*/*Real* token logits. We use `Qwen/Qwen3-VL-32B-Instruct` for this stage in all experiments.

### Run

The whole pipeline is driven by `main.py`, whose behavior is selected by the `task` field of the Hydra config. A complete run on a dataset (e.g., FakeTT) proceeds as follows:

```sh
# Stage 1: confidence-aware pseudo-annotation (local model, requires logits)
python main.py --config-name label_qwen_FakeTT model_id=Qwen/Qwen3-VL-32B-Instruct

# Stage 2: build embeddings and construct semantically aligned comparison groups
python model/Insight/make_embeddings.py
python model/Insight/conduct_retrieval.py

# Stage 3: discrepancy mining
python main.py --config-name insight_qwen_FakeTT

# Stage 3: heuristic set compilation
python main.py --config-name manage_qwen_FakeTT

# Stage 4: cue-guided inference
python main.py --config-name inpredict_qwen_FakeTT
```

Alternatively, a convenience script that reproduces the full pipeline on all three datasets is provided:

```sh
bash script/run_fsol_qwen.sh
```

## Backbones

FSoL is a backbone-agnostic, plug-and-play framework. In our experiments we evaluate it on six LVLMs: Qwen2.5-VL-7B/72B, LLaVA-v1.6-7B/34B, and Gemma3-4B/27B. The corresponding configs are provided for each backbone in `config/`. As noted above, the pseudo-annotation stage always uses a locally deployed model to access token-level logits.

## Prompts

For full transparency and reproducibility, the prompt used at each stage of the framework can be found at the following exact locations.

| Stage | Prompt location |
| --- | --- |
| Stage 1 — Confidence-Aware Pseudo-Annotation | the `prompt` field in [`config/label_qwen_FakeTT.yaml`](config/label_qwen_FakeTT.yaml#L6) (and the other `config/label_*.yaml`), assembled in [`model/Label/label_runner.py`](model/Label/label_runner.py#L285-L297) |
| Stage 2 — Comparison Group Construction | the pairing prompt in [`model/Insight/insight_runner.py`](model/Insight/insight_runner.py#L201) |
| Stage 3 — Discrepancy Mining | [`model/Insight/insight_runner.py`](model/Insight/insight_runner.py#L201) (optionally overridden by the `prompt` field in `config/insight_*.yaml`) |
| Stage 3 — Heuristic Compilation | the `prompt_template` in [`model/Manage/manage_runner.py`](model/Manage/manage_runner.py#L17) |
| Stage 4 — Cue-Guided Inference | the `prompt_template` in [`model/InPredict/inpredict_runner.py`](model/InPredict/inpredict_runner.py#L55) |
