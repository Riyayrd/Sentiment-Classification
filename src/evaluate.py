# src/evaluate.py
import os
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

from models import SentimentRNN

def load_processed(data_dir, seq_len):
    x_test = np.load(os.path.join(data_dir, f"X_test_{seq_len}.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return x_test, y_test

def evaluate(model, model_state_path, data_dir, seq_len, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_test, y_test = load_processed(data_dir, seq_len)
    x_test = torch.LongTensor(x_test)
    y_test = torch.LongTensor(y_test)

    test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # load model architecture from state naming convention: first line create model with provided args
    # For safety caller should recreate the right model and pass here.
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.extend((out.cpu().numpy() >= 0.5).astype(int).tolist())
            all_labels.extend(yb.numpy().astype(int).tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds)
    return acc, f1, report

def plot_history(history_path, out_png):
    df = pd.read_pickle(history_path)
    plt.figure()
    plt.plot(df['train_loss'], label='train_loss')
    plt.plot(df['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["RNN","LSTM","BiLSTM"], required=True)
    parser.add_argument("--model_state", type=str, required=True, help="path to .pt state dict")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, choices=[25,50,100], required=True)
    parser.add_argument("--history_path", type=str, required=False, help="path to history pickle to plot")
    args = parser.parse_args()

    model = SentimentRNN(model_type=args.model_type,
                         vocab_size=10000,
                         embed_dim=100,
                         hidden_size=64,
                         num_layers=2,
                         dropout=0.4,
                         activation='tanh')  # activation only affects classifier; replace if needed

    acc, f1, report = evaluate(model, args.model_state, args.data_dir, args.seq_len)
    print("Accuracy:", acc)
    print("F1 (macro):", f1)
    print("Classification report:\n", report)
    if args.history_path:
        out_png = os.path.join(os.path.dirname(args.history_path), os.path.basename(args.history_path).replace("_history.pkl","_loss.png"))
        plot_history(args.history_path, out_png)
        print("Saved loss plot to", out_png)
