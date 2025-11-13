# src/train.py
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from models import SentimentRNN
from utils import set_seed, device_info

def load_processed(data_dir, seq_len):
    x_train = np.load(os.path.join(data_dir, f"X_train_{seq_len}.npy"))
    x_test = np.load(os.path.join(data_dir, f"X_test_{seq_len}.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return x_train, y_train, x_test, y_test

def train_one(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test = load_processed(config["data_dir"], config["seq_len"])
    x_train = torch.LongTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.LongTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model = SentimentRNN(model_type=config["arch"],
                         vocab_size=config["vocab_size"],
                         embed_dim=config["embed_dim"],
                         hidden_size=config["hidden_size"],
                         num_layers=config["num_layers"],
                         dropout=config["dropout"],
                         activation=config["activation"]).to(device)

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        raise ValueError("Unknown optimizer")

    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [], "acc": [], "f1": [], "epoch_time": []}

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if config["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_val"])
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # validation on test set
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
                all_labels.extend(yb.cpu().numpy().astype(int).tolist())
        val_loss = val_loss / len(test_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(acc)
        history["f1"].append(f1)
        history["epoch_time"].append(epoch_time)

        print(f"[{config['name']}] Epoch {epoch}/{config['epochs']} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={acc:.4f} f1={f1:.4f} time={epoch_time:.2f}s")

    # Save model & history
    os.makedirs(config["save_dir"], exist_ok=True)
    model_path = os.path.join(config["save_dir"], config["name"] + ".pt")
    torch.save(model.state_dict(), model_path)

    hist_path = os.path.join(config["save_dir"], config["name"] + "_history.pkl")
    pd.DataFrame(history).to_pickle(hist_path)

    summary = {
        "Model": config["name"],
        "Architecture": config["arch"],
        "Activation": config["activation"],
        "Optimizer": config["optimizer"],
        "Seq Length": config["seq_len"],
        "Grad Clipping": "Yes" if config["grad_clip"] else "No",
        "Accuracy": history["acc"][-1],
        "F1": history["f1"][-1],
        "Epoch Time (s)": float(np.mean(history["epoch_time"])),
        "model_path": model_path,
        "history_path": hist_path,
        "device_info": device_info()
    }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="processed data directory (from preprocess.py)")
    parser.add_argument("--save_dir", type=str, default="results/models")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["RNN","LSTM","BiLSTM"], required=True)
    parser.add_argument("--activation", type=str, choices=["relu","tanh","sigmoid"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["Adam","SGD","RMSProp"], required=True)
    parser.add_argument("--seq_len", type=int, choices=[25,50,100], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--grad_clip", action="store_true")
    parser.add_argument("--grad_clip_val", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = vars(args)
    # fixed architecture params per assignment
    cfg.update({
        "vocab_size": 10000,
        "embed_dim": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": cfg.get("batch_size",32),
        "save_dir": args.save_dir
    })

    summary = train_one(cfg)
    # save summary as CSV/one-row file in save_dir
    import pandas as pd
    pd.DataFrame([summary]).to_csv(os.path.join(cfg["save_dir"], cfg["name"] + "_summary.csv"), index=False)
    print("Saved summary to", cfg["save_dir"])
