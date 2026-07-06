#!/bin/bash
# =============================================================================
# FSoL: Training-Free Self-Optimization with LVLMs for Zero-Shot FNVD
# Full pipeline on the target datasets (FakeTT / FakeSV / FVC).
# =============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Confidence-Aware Pseudo-Annotation
# Must use a LOCALLY deployed model to access the Fake/Real token logits.
# ----------------------------------------------------------------------------
python main.py --config-name label_qwen_FakeTT model_id=Qwen/Qwen3-VL-32B-Instruct
python main.py --config-name label_qwen_FVC    model_id=Qwen/Qwen3-VL-32B-Instruct
python main.py --config-name label_qwen_FakeSV model_id=Qwen/Qwen3-VL-32B-Instruct

# ----------------------------------------------------------------------------
# Stage 2: Semantically Aligned Comparison Group Construction
# Build embeddings and retrieve pseudo-fake / pseudo-real comparison pairs.
# ----------------------------------------------------------------------------
python model/Insight/make_embeddings.py
python model/Insight/conduct_retrieval.py

# ----------------------------------------------------------------------------
# Stage 3: Contrastive-Reasoning-Driven Self-Refinement
#   (a) discrepancy mining      -> task: insight
#   (b) heuristic compilation   -> task: manage
# ----------------------------------------------------------------------------
python main.py --config-name insight_qwen_FakeTT
python main.py --config-name manage_qwen_FakeTT

# ----------------------------------------------------------------------------
# Stage 4: Cue-Guided Inference  -> task: inpredict
# ----------------------------------------------------------------------------
python main.py --config-name inpredict_qwen_FakeTT
