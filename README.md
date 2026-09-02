# TIC-Bench

Official repository for the paper **"Deeply Interleaved Text-Image Contexts for Multimodal LLMs Assessment."**

TIC-Bench (deeply interleaved **T**ext-**I**mage **C**ontexts) is a benchmark for evaluating whether multimodal large language models can continuously bind, integrate, and propagate information distributed across long sequences of visual and textual evidence. This repository primarily provides the inference and evaluation code for the benchmark.

<p align="center">
  <img src="assets/ticbench_teaser.png" alt="TIC-Bench input comparison and benchmark composition" width="82%">
</p>

<p align="center"><em>TIC-Bench contrasts conventional multi-image inputs with deeply interleaved text-image contexts and contains 2,280 questions across three association domains.</em></p>

## Overview

Most existing multimodal benchmarks focus on a single image or treat multiple images as a parallel collection while text serves mainly as an instruction. TIC-Bench instead evaluates reasoning over densely interleaved text-image contexts, where neither modality alone is sufficient to recover the answer.

The benchmark contains **2,280 questions** across three complementary association domains and eight task types:

| Domain | Task type | Questions | What it evaluates |
| --- | --- | ---: | --- |
| Logical Association | Linear Logic | 260 | Following a single cross-image relation chain |
| Logical Association | Cyclic Logic | 249 | Revisiting earlier images during reasoning |
| Logical Association | Convergent Logic | 275 | Combining multiple independent evidence branches |
| Temporal Association | Sequential Scenario | 270 | Following events and states forward in time |
| Temporal Association | Retrospective Scenario | 226 | Retrieving evidence from earlier story segments |
| Temporal Association | Parallel Scenario | 230 | Coordinating multiple concurrent storylines |
| Spatial Association | Map | 385 | Reconstructing spatial relations in aerial imagery |
| Spatial Association | Photo | 385 | Reconstructing spatial relations in natural images |

Domain totals are 784 Logical, 726 Temporal, and 770 Spatial questions. Logical and Spatial Association use multiple-choice questions, while Temporal Association includes 418 open-ended questions.

## Benchmark Statistics

TIC-Bench is designed around long, densely interleaved multimodal contexts:

| Statistic | Value |
| --- | ---: |
| Questions | 2,280 |
| Image instances | 45,776 |
| Total image references | 85,145 |
| Images per question | 20.08 |
| Image references per question | 37.34 |
| References per image | 1.86 |
| Average textual context length | 309.3 words |

## Task Domains

<p align="center">
  <img src="assets/ticbench_overview.png" alt="Examples of Spatial, Logical, and Temporal Association tasks" width="100%">
</p>

<p align="center"><em>Representative deeply interleaved contexts from Spatial, Logical, and Temporal Association.</em></p>

### Logical Association

Logical instances contain ordered scene images connected through shared objects and relations. Questions identify target objects indirectly through cross-image descriptions, requiring models to ground textual references in visual entities and propagate those bindings through Linear, Cyclic, or Convergent reasoning structures.

### Temporal Association

Temporal instances use comic-style storyboards with interleaved descriptions. Images provide character identities, scene states, and event details, while text provides actions, causal relations, and plot progression. Character names and selected event information are masked or generalized to reduce text-only shortcuts.

### Spatial Association

Spatial instances divide a large source image into overlapping patches. Selected patches are replaced with textual descriptions of their content and spatial relationships to neighboring patches. Models must combine visible crops, descriptions of missing regions, and target views to infer relative positions in map-based or natural-image environments.

## Evaluation Protocol

The paper evaluates five open-source MLLMs in both standard and thinking modes:

- Gemma-4-31B
- GLM-4.6V
- Qwen3.6-35B-A3B
- Kimi-K2.6
- MiMo-V2.5

It also evaluates five closed-source MLLMs using their default inference settings:

- Claude Sonnet 4.6
- Gemini 3.1 Pro Preview
- GPT-5.5
- Doubao-Seed-2.0-Pro
- GLM-5V-Turbo

This produces 15 model inference settings in total. The paper uses Qwen 3.7 Plus, DeepSeek V4 Pro, and Claude Opus 4.8 as three independent automatic judges. Cases with inconsistent judgments are resolved through human evaluation.

The released evaluation code preserves the three independent judgments, applies a 2-out-of-3 majority vote, records agreement and unanimity, and exposes disagreements for subsequent human review. Each judge evaluates the model's final conclusion and returns:

```json
{
  "consistent": true,
  "correct": true,
  "score": 100,
  "reason": "Concise justification"
}
```

## Main Findings

The strongest evaluated model, GPT-5.5, reaches **59.9%** overall accuracy, compared with **91.7%** for human experts. Gemini 3.1 Pro Preview reaches **59.0%**. The gap of more than 30 percentage points shows that current MLLMs still struggle to maintain and combine evidence distributed across interleaved images and text.

Convergent logical reasoning and Parallel temporal reasoning are especially challenging. The paper's error analysis further identifies abstraction, reasoning, and hallucination errors as major failure modes, particularly when models must maintain stable entity mappings or merge multiple evidence streams.

## Repository Scope

This public repository focuses on dataset evaluation and includes:

- Dataset loading for all three association domains
- Multimodal input assembly
- Resumable model inference
- Full, Text-only, and Images-only input settings
- Three-judge semantic evaluation and majority voting
- Accuracy, score, agreement, and unanimity reports
- Tests for prompt safety, result preservation, and judge recovery
- One minimal, replaceable OpenAI Responses client template

The repository does not include dataset-construction pipelines, image-generation code, model weights, service endpoints, credentials, usage tracking, or request logs.

## Installation

Python 3.10 or later is required.

```bash
python -m pip install -e ".[test]"
```

The client uses the standard runtime configuration supported by the official SDK. No credential or endpoint value is stored in this repository.

## Dataset Layout

Each benchmark domain is organized into `question_*` directories.

```text
DATASET_ROOT/
├── question_0/
│   ├── benchmark question data
│   ├── referenced images
│   ├── answer_results_<model>.json
│   └── evaluation_results_<model>.json
├── question_1/
└── ...
```

The loaders recognize the following domain-specific files:

| Domain | Question data |
| --- | --- |
| Logical | `qa_pairs*.json` and `node_sequence*.json` |
| Temporal | `qa_pairs*.json` with a top-level `qa` list |
| Spatial | `dataset_caption.jsonl` |

In the command-line interface, Logical Association uses the internal kind name `visual` for compatibility with the original data layout.

The released dataset is identified as `pino10010/TIC-Bench` on Hugging Face.

## Usage

### Validate Input Assembly

Use dry-run mode to validate dataset discovery and message construction without sending model requests:

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --dry-run
```

Specify the domain explicitly when automatic detection is not appropriate:

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --kind spatial --dry-run
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --kind visual --dry-run
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --kind temporal --dry-run
```

### Run Model Inference

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME
```

Available input settings are:

- `full`: retain the complete interleaved text-image context
- `text_only`: remove image contents while preserving textual reference markers and question text
- `images_only`: remove descriptive textual evidence while retaining the images and question
- `question_images_only`: retain question-related images and the question text

Example:

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --mode text_only
```

Inference is resumable. Any existing non-empty answer is preserved regardless of its historical status field, and historical records outside the current dataset view are not discarded.

### Run Three-Judge Evaluation

`TARGET_NAME` is the portion of an answer filename between `answer_results_` and `.json`.

```bash
python -m ticbench evaluate DATASET_ROOT \
  --judge-model JUDGE_1 JUDGE_2 JUDGE_3 \
  --target TARGET_NAME
```

Three distinct judge names are required. Each completed evaluation stores the individual judge outputs together with:

- `majority_vote`
- `judge_votes`
- `judge_consistency_votes`
- `agreement_count`
- `unanimous`
- mean `score`

If one or more judge calls fail, successful judgments are retained and a later run requests only the missing judgments.

### Summarize Results

```bash
python -m ticbench report DATASET_ROOT
```

The report contains the number of evaluated questions, correct answers, accuracy, mean score, unanimous decisions, and unanimity rate for each evaluated model.

## Code Structure

```text
ticbench/
├── cli.py                 # Command-line interface
├── questions.py           # Domain loaders and multimodal input assembly
├── inference.py           # Resumable answer generation
├── evaluation.py          # Three-judge evaluation and aggregation
├── report.py              # Offline result summaries
├── io.py                  # Dataset detection and safe JSON utilities
└── openai_template.py     # The single model-service adapter template

tests/
├── test_questions.py      # Prompt construction and answer-leakage tests
├── test_inference.py      # Resume and answer-preservation tests
└── test_evaluation.py     # Three-judge and recovery tests
```

## Authors

Zihao Wang, Xi Xiang, Yuwen Sun, Yingyu Li, Yabo Zhang, Yihan Zeng, Fan Li, and Wangmeng Zuo.

Harbin Institute of Technology and Huawei Noah's Ark Lab.

## Citation

Please cite **"Deeply Interleaved Text-Image Contexts for Multimodal LLMs Assessment"** when using TIC-Bench. Complete publication metadata and BibTeX will be added with the public paper release.
