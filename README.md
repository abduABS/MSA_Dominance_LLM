# CMS-LDP-PRO

**Paper context:** _"Does Modern Standard Arabic (MSA) Dominate Arabic Dialects in LLMs? A Representation-Level Analysis."_

This repository contains the datasets, notebooks, and scripts used to support analysis of Arabic dialect representations in large language models (LLMs). The work investigates whether MSA acts as a dominant internal representation for dialectal Arabic, and finds that dialectal representations form a dense, overlapping space rather than an MSA-centric one.

## Key contributions

- Adapts language dominance probing to a fine-grained Arabic dialect setting with 26 varieties plus MSA.
- Investigates representation geometry across model layers and architecture families.
- Separates generation bias from internal representational dominance.
- Supports experiments for model families including BLOOM, mGPT, LLaMA, Qwen, and ArabianGPT.

## Contents

- `ArabianGPT_Dialect.ipynb` - Dialect-related experiments using ArabianGPT.
- `BLOOM_PUD.ipynb` - PUD dataset evaluation and analysis using BLOOM.
- `Llama_Dialect.ipynb` - Dialect experiments using LLaMA-based prompts and representations.
- `mGPT_Dialect.ipynb` - Dialect representation experiments with mGPT.
- `mGPT_PUD.ipynb` - Partial utterance dataset (PUD) evaluation using mGPT.
- `Qwen_Dialect.ipynb` - Dialectal analysis with Qwen models.
- `MADAR.combined.Arabic.csv` - Core Arabic dialect dataset for analysis.
- `multilingual_pud_dataset_v2.17.csv` - Partial utterance dataset used for PUD experiments.

### Analysis and utility scripts

- `Dialect Similarity/dialect_similarity.py` - Evaluates pairwise dialect similarity using:
  - character n-gram TF-IDF + cosine similarity
  - word-level Jaccard similarity
  - sentence-level word-overlap similarity
  - ensemble scoring and heatmap visualizations

- `POS Analysis/arabic_pos_analysis.py` - Rule-based Arabic POS tagging, filler removal, and analysis of function-word distributions.

- `POS Percentage/code.py` - Generates CSV variants with 20%, 40%, and 60% of each sentence removed from the start of non-ID columns, supporting partial utterance studies.

- `POS Percentage/MADAR.combined.Arabic_20percent.csv`
- `POS Percentage/MADAR.combined.Arabic_40percent.csv`
- `POS Percentage/MADAR.combined.Arabic_60percent.csv`

## Installation

Install the required Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter notebooks

Open the project notebooks with Jupyter Lab, Notebook, or VS Code to reproduce model analysis and visualization workflows.

```bash
jupyter lab
```

### Scripts

#### Dialect similarity analysis

```bash
python "Dialect Similarity/dialect_similarity.py"
```

This script computes similarity matrices between dialects and saves heatmaps and ranking outputs.

#### POS analysis

```bash
python "POS Analysis/arabic_pos_analysis.py"
```

This script runs rule-based POS tagging and filler analysis on Arabic text.

#### Generate partial utterance datasets

```bash
python "POS Percentage/code.py"
```

This generates new CSV files with 20%, 40%, and 60% of each sentence removed from the beginning of non-ID columns.

## Paper alignment

This repo is aligned with the paper’s core research goals:

- measuring whether MSA internally dominates dialectal Arabic representations in LLMs
- comparing representation geometry across dialects and model layers
- analyzing how dense overlap among dialect representations impacts model behavior
- separating output preference from internal representational structure

## Notes

- Paths in notebook code may need adjustment for your local filesystem.
- The dataset files are large, so ensure sufficient memory when loading with pandas.
- Notebooks are the primary interface for reproducing paper experiments.

## Requirements

- Python 3.10+ recommended
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `torch`
- `transformers`
- `regex`
- `tqdm`
- `ipython`
- `ipytest`
- `pytest`

---

This repository supports Arabic representation-level research for dialect analysis, model comparisons, and partial utterance dataset evaluation.
