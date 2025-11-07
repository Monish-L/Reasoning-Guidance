# Enhancing Medical Reasoning in Small Language Models Through Feedback-Guided Refinement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Master's Thesis** | San José State University | Department of Computer Engineering  
> **Author:** Monish Sai Lakamraju  
> **Advisors:** Dr. KaiKai Lui, Dr. Bernardo Flores, Dr. Mahima Agumbe Suresh  
> **Defense Date:** December 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Studies](#ablation-studies)
- [Computational Requirements](#computational-requirements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This repository contains the complete implementation of **Feedback-Guided Refinement Fine-Tuning (FGRFT)**, a novel approach to enhance medical reasoning in Small Language Models (< 10B parameters) using minimal training data.

### The Problem

- Large Language Models (>100B parameters) excel at medical reasoning but are computationally prohibitive
- Small Language Models (<10B) are deployable in resource-constrained settings but struggle with multi-step clinical reasoning


### Our Solution

A **five-stage pipeline** that:
1. Generates answer-agnostic reasoning (GPT-4o)
2. Verifies and provides structured feedback (GPT-5 as judge)
3. Applies preservation-aware corrections
4. Validates with single-pass gate
5. Transforms into conversational format

**Result:** Achieves 68.7% on MedQA-USMLE with only **5,000 training samples** 

---

## 🏆 Key Contributions

1. **Data-Efficient Pipeline**: Requires only 5k samples vs. 20k-32k in comparable methods
2. **Preservation-Aware Refinement**: Maintains correct reasoning while surgically correcting errors
3. **Strong Performance**: 68.7% MedQA accuracy (+9.7pp over base), competitive with methods using 4-6× more data
4. **Temperature Robustness**: Minimal degradation (-0.2pp) across sampling strategies (τ=0.5 to τ=0.7)
5. **Verified Ablation**: +4.2pp improvement over naive CoT under controlled conditions

---

## 📊 Results Summary

### Main Results (τ = 0.5)

| Model | MedQA | PubMedQA | MedMCQA | Overall |
|-------|-------|----------|---------|---------|
| **FGRFT (Ours)** | **68.7%** | **77.3%** | **57.7%** | **68.0%** |
| Llama-3.1-8B (Base) | 58.9% | 76.8% | 53.6% | 62.8% |
| Llama-3-Med42-8B | 61.4% | 77.1% | 56.8% | 64.8% |
| Qwen-2.5-7B | 57.9% | 72.7% | 57.1% | 62.2% |
| BioMistral-7B | 41.6% | 29.1% | 40.7% | 37.5% |


### Ablation Study

| Method | Supervision | Samples | MedQA | Δ vs Base |
|--------|-------------|---------|-------|-----------|
| **FGRFT** | Verifier-audited | 5,000 | **68.7%** | **+9.7pp** |
| Naive CoT | Unverified | 5,000 | 64.5% | +5.6pp |
| **Δ (FGRFT - Naive)** | — | — | **+4.2pp** | — |

---

## 📁 Repository Structure
```
thesis-fgrft/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── configs/
│   └── deepspeed_zero3.yaml          # DeepSpeed ZeRO-3 configuration
├── data/
│   ├── training_5k_sgft.json         # FGRFT training data (5k)
│   ├── training_5k_mixtral_naive_cot.json  # Naive CoT baseline
│   ├── eval_data_clinical.json       # MedQA + PubMedQA test set
│   └── eval_data_medmcqa.json        # MedMCQA validation set
├── stage1_data_generation/
│   ├── prompt_templates/
│   │   ├── P1_reasoning_generator.txt
│   │   ├── P2_verifier_judge.txt
│   │   ├── P3_guided_corrector.txt
│   │   ├── P2.5_validation_gate.txt
│   │   ├── P4_conversational_cot.txt
│   │   └── P5_final_response.txt
│   ├── generate_reasoning_data.py    # Stage 1 pipeline implementation
│   └── utils/
│       ├── api_clients.py            # GPT-4o/GPT-5 API wrappers
│       └── data_processing.py        # JSON/JSONL utilities
├── stage2_finetuning/
│   ├── finetuning_sgft.py            # FGRFT supervised fine-tuning
│   ├── finetuning_naive_cot.py       # Naive CoT baseline training
│   └── utils/
│       ├── chat_templates.py         # Llama-3 chat formatting
│       └── data_collators.py         # Custom data collators
├── evaluation/
│   ├── eval.py                       # Main evaluation script
│   ├── accuracy.py                   # Accuracy computation
│   └── vllm_server.py                # vLLM inference server wrapper
├── scripts/
│   ├── train_sgft.sh                 # Training launch script
│   ├── train_naive_cot.sh            # Naive CoT training script
│   ├── eval_all_benchmarks.sh        # Evaluation script
│   └── run_ablation.sh               # Ablation study runner
├── results/
│   ├── sgft_results/                 # FGRFT evaluation outputs
│   ├── naive_cot_results/            # Naive CoT evaluation outputs
│   └── ablation_results/             # Ablation study results
├── models/
│   ├── fine-tuned-5k/                # Final FGRFT model checkpoint
│   └── naive-cot-5k/                 # Naive CoT model checkpoint
└── docs/
    ├── thesis.pdf                    # Full thesis document
    ├── pipeline_diagram.png          # Architecture diagram
    └── supplementary.pdf             # Additional materials
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 8× NVIDIA A100 80GB GPUs (or equivalent)

### Setup
```bash
# Clone repository
git clone https://github.com/monishlakamraju/thesis-fgrft.git
cd thesis-fgrft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install flash-attn --no-build-isolation
pip install deepspeed
```

### Requirements.txt
```txt
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
deepspeed>=0.12.0
vllm>=0.2.0
openai>=1.0.0
wandb>=0.16.0
datasets>=2.14.0
sentencepiece>=0.1.99
protobuf>=3.20.0
jinja2>=3.1.2
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## 🚀 Quick Start

### 1. Download Base Model
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/Llama-3.1-8B-Instruct \
    --cache-dir ~/.cache/huggingface
```

### 2. Prepare Training Data
```bash
# Our curated FGRFT data (5k samples)
cp data/training_5k_sgft.json stage2_finetuning/

# Or generate your own using Stage 1 pipeline
python stage1_data_generation/generate_reasoning_data.py \
    --input_file data/medqa_train.json \
    --output_file data/custom_fgrft_5k.json \
    --num_samples 5000
```

### 3. Train Model
```bash
# FGRFT training
bash scripts/train_sgft.sh

# Or manually:
accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    stage2_finetuning/finetuning_sgft.py \
    --model_path models/Llama-3.1-8B-Instruct \
    --data_path data/training_5k_sgft.json \
    --output_dir ckpts/sgft_finetuning \
    --n_epochs 3 \
    --learning_rate 5e-6 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_len 8192
```

### 4. Evaluate
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ckpts/sgft_finetuning/checkpoint-2-936/tfmr \
    --port 30000 \
    --tensor-parallel-size 8

# Run evaluation
python evaluation/eval.py \
    --model_name ckpts/sgft_finetuning/checkpoint-2-936/tfmr \
    --eval_file data/eval_data_clinical.json \
    --port 30000 \
    --max_new_tokens 2000 \
    --batch_size 256
```

---

## 🔄 Pipeline Stages

### Stage 1: Feedback-Guided Data Generation

**Objective:** Generate 5k high-quality reasoning chains with verifier feedback
```python
# Example usage
from stage1_data_generation.generate_reasoning_data import FGRFTPipeline

pipeline = FGRFTPipeline(
    generator_model="gpt-4o",
    verifier_model="gpt-5",
    num_samples=5000
)

# Generate enhanced dataset
enhanced_data = pipeline.run(
    input_file="data/medqa_train.json",
    output_file="data/training_5k_fgrft.json"
)
```

**Pipeline Flow:**
```
Question → P1: Answer-agnostic reasoning (GPT-4o)
         ↓
         P2: Verifier audit (GPT-5 + gold answer)
         ↓
         P3: Guided correction (GPT-4o + feedback)
         ↓
         P2.5: Validation gate (Pass/Revise, 1 loop max)
         ↓
         P4: Conversational CoT formatting
         ↓
         P5: Final clinical response
         ↓
         Enhanced dataset (5k samples)
```

### Stage 2: Supervised Fine-Tuning

**Objective:** Transfer reasoning capability to Llama-3.1-8B
```bash
accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    stage2_finetuning/finetuning_sgft.py \
    --model_path models/Llama-3.1-8B-Instruct \
    --data_path data/training_5k_sgft.json \
    --output_dir ckpts/sgft_finetuning \
    --n_epochs 3 \
    --max_seq_len 8192 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_rates 0.05 \
    --weight_decay 0.1
```

**Training Details:**
- Optimizer: AdamW (lr=5×10⁻⁶, weight decay=0.1)
- Schedule: Cosine decay with 5% warmup
- Precision: BFloat16
- Parallelism: DeepSpeed ZeRO-3
- Effective batch size: 128 (2 × 8 GPUs × 8 accum steps)
- Training time: ~16 minutes on 8× A100 80GB

---

## 📊 Datasets

### Training Data

| Dataset | Source | Samples | Format |
|---------|--------|---------|--------|
| **FGRFT (Ours)** | Stage 1 pipeline | 5,000 | Verifier-audited CoT |
| **Naive CoT** | HPAI-BSC/MedQA-Mixtral-CoT | 5,000 | Mixtral-generated CoT |

### Evaluation Data

| Benchmark | Split | Samples | Task Type |
|-----------|-------|---------|-----------|
| **MedQA-USMLE** | Test | 1,273 | 4-choice clinical vignettes |
| **PubMedQA** | Test | 1,000 | 3-choice evidence QA |
| **MedMCQA** | Val | 1,000 | 4-choice medical exams |

---

## 🏋️ Training

### FGRFT Training
```bash
#!/bin/bash
# scripts/train_sgft.sh

export WANDB_PROJECT="thesis-fgrft"
export HF_TOKEN="your_huggingface_token"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    stage2_finetuning/finetuning_sgft.py \
    --experiment_name sgft_finetuning \
    --model_path models/Llama-3.1-8B-Instruct \
    --data_path data/training_5k_sgft.json \
    --output_dir ckpts \
    --log_dir train_logs \
    --n_epochs 3 \
    --max_seq_len 8192 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_rates 0.05 \
    --weight_decay 0.1 \
    --max_ckpts 2 \
    --seed 42
```

### Naive CoT Baseline Training
```bash
#!/bin/bash
# scripts/train_naive_cot.sh

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    stage2_finetuning/finetuning_naive_cot.py \
    --experiment_name naive_cot_finetuning \
    --model_path models/Llama-3.1-8B-Instruct \
    --data_path data/training_5k_mixtral_naive_cot.json \
    --output_dir ckpts \
    --n_epochs 3 \
    --max_seq_len 8192 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_rates 0.05 \
    --max_ckpts 2
```

### DeepSpeed Configuration
```yaml
# configs/deepspeed_zero3.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

---

## 🧪 Evaluation

### Start Inference Server
```bash
# Start vLLM server (Terminal 1)
python -m vllm.entrypoints.openai.api_server \
    --model ckpts/sgft_finetuning/checkpoint-2-936/tfmr \
    --port 30000 \
    --tensor-parallel-size 8 \
    --max-model-len 8192
```

### Run Evaluation
```bash
# Clinical benchmarks (Terminal 2)
python evaluation/eval.py \
    --model_name ckpts/sgft_finetuning/checkpoint-2-936/tfmr \
    --eval_file data/eval_data_clinical.json \
    --port 30000 \
    --max_new_tokens 2000 \
    --batch_size 256

# MedMCQA
python evaluation/eval.py \
    --model_name ckpts/sgft_finetuning/checkpoint-2-936/tfmr \
    --eval_file data/eval_data_medmcqa.json \
    --port 30000 \
    --max_new_tokens 2000 \
    --batch_size 256
```

### Evaluation Script
```bash
#!/bin/bash
# scripts/eval_all_benchmarks.sh

MODEL_PATH="ckpts/sgft_finetuning/checkpoint-2-936/tfmr"
PORT=30000

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $PORT \
    --tensor-parallel-size 8 &

# Wait for server to start
sleep 120

# Run all evaluations
for EVAL_FILE in data/eval_data_clinical.json data/eval_data_medmcqa.json; do
    python evaluation/eval.py \
        --model_name $MODEL_PATH \
        --eval_file $EVAL_FILE \
        --port $PORT \
        --max_new_tokens 2000 \
        --batch_size 256
done

# Kill server
pkill -f vllm
```

---

## 🔬 Ablation Studies

### 1. FGRFT vs Naive CoT
```bash
#!/bin/bash
# scripts/run_ablation.sh

# Train naive CoT baseline
bash scripts/train_naive_cot.sh

# Evaluate both models
for MODEL in sgft_finetuning naive_cot_finetuning; do
    python -m vllm.entrypoints.openai.api_server \
        --model ckpts/$MODEL/checkpoint-2-936/tfmr \
        --port 30000 \
        --tensor-parallel-size 8 &
    
    sleep 120
    
    python evaluation/eval.py \
        --model_name ckpts/$MODEL/checkpoint-2-936/tfmr \
        --eval_file data/eval_data_clinical.json \
        --port 30000 \
        --max_new_tokens 2000 \
        --batch_size 256
    
    pkill -f vllm
done
```

**Results:**

| Method | MedQA | Δ vs Base | Δ vs Naive |
|--------|-------|-----------|------------|
| FGRFT | 68.7% | +9.7pp | +4.2pp |
| Naive CoT | 64.5% | +5.6pp | — |

### 2. Temperature Sensitivity
```python
# evaluation/temperature_sensitivity.py

TEMPERATURES = [0.5, 0.7]
MODEL_PATH = "ckpts/sgft_finetuning/checkpoint-2-936/tfmr"

for temp in TEMPERATURES:
    results = evaluate_model(
        model_path=MODEL_PATH,
        eval_file="data/eval_data_clinical.json",
        temperature=temp,
        top_p=0.9
    )
    print(f"Temperature {temp}: MedQA={results['medqa']:.1f}%, PubMedQA={results['pubmedqa']:.1f}%")
```

**Results:**

| Temperature | MedQA | PubMedQA | Average |
|-------------|-------|----------|---------|
| 0.5 | 68.7% | 77.3% | 73.0% |
| 0.7 | 67.6% | 77.9% | 72.8% |
| Δ | -1.1pp | +0.6pp | -0.2pp |

---

## 💻 Computational Requirements

### Training

- **Hardware:** 8× NVIDIA A100 80GB GPUs
- **Memory:** ~640GB GPU RAM (ZeRO-3 sharding)
- **Time:** ~16 minutes per training run
- **Cost:** ~$30 on Lambda Labs

### Inference

- **Hardware:** 8× NVIDIA A100 80GB GPUs
- **Memory:** ~60GB GPU RAM (tensor parallelism)
- **Throughput:** ~250 samples/minute (batch size 256)
- **Latency:** ~2 seconds per sample

### Stage 1 Data Generation

- **API Calls:** ~30k (5k samples × 6 prompts)
- **Cost:** ~$150 (GPT-4o + GPT-5)
- **Time:** ~4 hours

---

## 📝 Citation

If you use this code or findings in your research, please cite:
```bibtex
@mastersthesis{lakamraju2025fgrft,
  title={Enhancing Medical Reasoning in Small Language Models Through Feedback-Guided Refinement},
  author={Lakamraju, Monish Sai},
  year={2025},
  school={San Jos\'{e} State University},
  department={Department of Computer Engineering},
  address={San Jos\'{e}, California},
  month={December}
}
```

---



<p align="center">
  <strong>Built with ❤️ at San José State University</strong>
</p>
