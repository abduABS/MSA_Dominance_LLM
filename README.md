# Arabic Dialect Representation in LLMs

Code and data for the paper: *"Does Modern Standard Arabic (MSA) 
Dominate Arabic Dialects in LLMs? A Representation-Level Analysis."*

This repository contains the notebooks and scripts used to investigate 
whether MSA acts as a dominant internal representation for dialectal 
Arabic across 26 varieties in multilingual LLMs.

## Repository Structure

### Notebooks
Each notebook corresponds to a model evaluated in the paper:
- `BLOOM_PUD.ipynb` / `mGPT_PUD.ipynb` — Replication experiments
- `BLOOM_Dialect.ipynb` / `mGPT_Dialect.ipynb` — Main dialect analysis
- `Llama_Dialect.ipynb`, `Qwen_Dialect.ipynb`, 
  `ArabianGPT_Dialect.ipynb` — Additional model evaluations

### Scripts
- `Dialect Similarity/dialect_similarity.py` — Pairwise dialect 
  similarity analysis
- `POS Analysis/arabic_pos_analysis.py` — Rule-based Arabic POS 
  tagging and function-word filtering
- `POS Percentage/code.py` — Generates sentence-prefix removal 
  variants (20%, 40%, 60%)

### Data
- `MADAR.combined.Arabic.csv` — Arabic dialect dataset (26 varieties)
- `multilingual_pud_dataset_v2.17.csv` — Multilingual replication data

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Notes
- Python 3.10+ recommended
- Notebook paths may need adjustment for your local filesystem
- Large dataset files require sufficient available memory
