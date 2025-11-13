# run_experiments_limited.py
import itertools
import argparse
import pandas as pd
import os
from src.train import train_one

def make_config(name, data_dir, arch, activation, optimizer, seq_len, grad_clip, epochs, lr, save_dir):
    return {
        "name": name,
        "data_dir": data_dir,
        "arch": arch,
        "activation": activation,
        "optimizer": optimizer,
        "seq_len": seq_len,
        "grad_clip": grad_clip,
        "grad_clip_val": 1.0,
        "epochs": epochs,
        "batch_size": 32,
        "lr": lr,
        "seed": 42,
        "vocab_size": 10000,
        "embed_dim": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.4,
        "save_dir": save_dir
    }

def main(data_dir, save_dir, epochs):
    # Select subset of hyperparameters to get ~42 experiments
    seq_lengths = [25,50,100]              # 3
    archs = ['RNN','LSTM','BiLSTM']        # 3
    activations = ['relu','tanh']          # 2
    optimizers = ['Adam','SGD','RMSProp']  # 3
    grad_options = [False, True]           # 2

    # This gives 3*3*2*3*2 = 108, we will filter to ~42 by limiting seq_len combos
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # Generate full grid
    combos = list(itertools.product(archs, activations, optimizers, seq_lengths, grad_options))

    # Keep only ~42 experiments using a filter (example: skip some seq_len=100)
    filtered_combos = [c for c in combos if not (c[3]==100 and c[0]=='RNN')]  # filter 1/3 of 100-length RNN
    filtered_combos = filtered_combos[:42]  # enforce exactly 42 if needed

    for idx, combo in enumerate(filtered_combos, 1):
        arch, activation, optimizer, seq_len, grad = combo
        name = f"{arch}_{activation}_{optimizer}_seq{seq_len}_{'clip' if grad else 'noclip'}"
        print(f"Experiment {idx}/{len(filtered_combos)}: Running {name}")
        cfg = make_config(name, data_dir, arch, activation, optimizer, seq_len, grad, epochs, 1e-3, save_dir)
        summary = train_one(cfg)
        results.append(summary)
        pd.DataFrame(results).to_csv(os.path.join(save_dir, "metrics_partial.csv"), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    print("Saved final metrics to", os.path.join(save_dir, "metrics.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/imdb/processed")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.epochs)
