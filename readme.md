# Comparative Analysis of RNN Architectures for Sentiment Classification

## Overview
This repository implements multiple RNN-based architectures (RNN, LSTM, BiLSTM) to perform binary sentiment classification on the IMDb dataset.
It systematically evaluates variations: activation functions (relu,tanh,sigmoid), optimizers (Adam,SGD,RMSProp), sequence lengths (25,50,100), and gradient clipping (Yes/No).

## Repo structure
├── data/
│   └── processed_seq25/ processed_seq50/ processed_seq100/   # created by preprocess.py
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── results/
│   ├── models/
│   ├── history/
│   └── metrics.csv
├── run_experiments.py
├── requirements.txt
└── README.md

## Setup
1. Create a python venv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare Data
    ```python src/preprocess.py --seq_len 50 --out_dir data/processed_seq50
    ```

Repeat for seq_len 25 and 100 or use the driver (below)

Quick Dev Run
# Create all three processed datasets
python run_experiments.py --epochs 1   # this will invoke preprocess and run a small subset quickly (set epochs low for dev)


3. Full Experiment

```python run_experiments.py --epochs 5
```

Adjust epochs to your budget. For final results, set epochs to 5-10 (longer gives better models).