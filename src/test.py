# src/test.py
"""
Comprehensive results plotting & summary script.

Run from project root after you've run experiments:
    python src/test.py

Outputs:
- results/summary.csv
- results/plots/acc_f1_vs_seq.png
- results/plots/acc_f1_box_by_seq.png
- results/plots/acc_by_arch_seq.png
- results/plots/f1_heatmap_arch_seq.png
- results/plots/loss_<modelname>.png  (for each history pickle found under results/models/)
- prints top/bottom models by F1 and Accuracy
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config & paths ---
METRICS_CSV = "results/metrics.csv"
PLOTS_DIR = "results/plots"
MODELS_DIR = "results/models"  # where run_experiments saved model histories
SUMMARY_CSV = "results/summary.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load metrics ---
if not os.path.isfile(METRICS_CSV):
    print(f"ERROR: metrics CSV not found at '{METRICS_CSV}'. Run experiments first.")
    sys.exit(1)

df = pd.read_csv(METRICS_CSV)
if df.empty:
    print("ERROR: metrics.csv is empty.")
    sys.exit(1)

# Normalize column names if necessary
df.columns = [c.strip() for c in df.columns]

# Ensure expected numeric columns exist
for col in ["Accuracy", "F1", "Seq Length"]:
    if col not in df.columns:
        print(f"ERROR: Expected column '{col}' in metrics.csv")
        sys.exit(1)

# Convert numeric columns properly
df["Seq Length"] = df["Seq Length"].astype(int)
df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
df["F1"] = pd.to_numeric(df["F1"], errors="coerce")

# --- Console summaries ---
print("\n=== Dataset of experiment results ===")
print(f"Total experiments: {len(df)}")
print("Unique architectures:", df.get("Architecture", df.get("Model", "")).unique())
print("Sequence lengths tested:", sorted(df["Seq Length"].unique()))
print()

# Top/bottom by F1
print("Top 5 models by F1:")
print(df.sort_values("F1", ascending=False).head(5)[["Model","Architecture","Activation","Optimizer","Seq Length","F1","Accuracy"]].to_string(index=False))
print("\nBottom 5 models by F1:")
print(df.sort_values("F1", ascending=True).head(5)[["Model","Architecture","Activation","Optimizer","Seq Length","F1","Accuracy"]].to_string(index=False))

# Top/bottom by Accuracy
print("\nTop 5 models by Accuracy:")
print(df.sort_values("Accuracy", ascending=False).head(5)[["Model","Architecture","Activation","Optimizer","Seq Length","Accuracy","F1"]].to_string(index=False))
print("\nBottom 5 models by Accuracy:")
print(df.sort_values("Accuracy", ascending=True).head(5)[["Model","Architecture","Activation","Optimizer","Seq Length","Accuracy","F1"]].to_string(index=False))

# Save a compact summary CSV grouped by (Architecture, Seq Length)
group_cols = ["Architecture", "Seq Length"]
if "Architecture" in df.columns:
    summary = df.groupby(group_cols).agg({"Accuracy": ["mean","std"], "F1": ["mean","std"], "Epoch Time (s)": "mean"}).reset_index()
    # flatten columns
    summary.columns = [' '.join(filter(None, col)).strip() for col in summary.columns.values]
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to {SUMMARY_CSV}")
else:
    print("\nWarning: 'Architecture' column missing; skipping grouped summary CSV.")
    summary = None

# --- Plot 1: Accuracy & F1 (mean) vs Sequence Length (line plot) ---
mean_df = df.groupby("Seq Length")[["Accuracy","F1"]].mean().reset_index()
plt.figure()
plt.plot(mean_df["Seq Length"], mean_df["Accuracy"], marker='o', label='Accuracy')
plt.plot(mean_df["Seq Length"], mean_df["F1"], marker='o', label='F1 (macro)')
plt.xlabel("Sequence Length")
plt.xticks(mean_df["Seq Length"])
plt.title("Mean Accuracy & F1 vs Sequence Length")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
out1 = os.path.join(PLOTS_DIR, "acc_f1_vs_seq.png")
plt.tight_layout()
plt.savefig(out1)
plt.close()
print("Saved:", out1)

# --- Plot 2: Accuracy & F1 distribution by Sequence Length (boxplots) ---
plt.figure()
# boxplot requires list of arrays
acc_groups = [df.loc[df["Seq Length"]==s, "Accuracy"].dropna().values for s in sorted(df["Seq Length"].unique())]
f1_groups  = [df.loc[df["Seq Length"]==s, "F1"].dropna().values for s in sorted(df["Seq Length"].unique())]

# Accuracy boxplot
plt.subplot(2,1,1)
plt.boxplot(acc_groups, labels=sorted(df["Seq Length"].unique()))
plt.ylabel("Accuracy")
plt.title("Accuracy distribution by Sequence Length")

# F1 boxplot
plt.subplot(2,1,2)
plt.boxplot(f1_groups, labels=sorted(df["Seq Length"].unique()))
plt.ylabel("F1 (macro)")
plt.xlabel("Sequence Length")
plt.title("F1 distribution by Sequence Length")
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "acc_f1_box_by_seq.png")
plt.savefig(out2)
plt.close()
print("Saved:", out2)

# --- Plot 3: Mean Accuracy by Architecture and Sequence Length (grouped bar chart) ---
if "Architecture" in df.columns:
    pivot_acc = df.pivot_table(index="Seq Length", columns="Architecture", values="Accuracy", aggfunc="mean")
    pivot_f1  = df.pivot_table(index="Seq Length", columns="Architecture", values="F1", aggfunc="mean")

    # Plot Accuracy grouped bars
    ax = pivot_acc.plot(kind='bar', rot=0, figsize=(8,5))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Mean Accuracy by Architecture and Sequence Length")
    plt.tight_layout()
    out3 = os.path.join(PLOTS_DIR, "acc_by_arch_seq.png")
    plt.savefig(out3)
    plt.close()
    print("Saved:", out3)

    # Plot F1 grouped bars
    ax = pivot_f1.plot(kind='bar', rot=0, figsize=(8,5))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean F1 (macro)")
    ax.set_title("Mean F1 by Architecture and Sequence Length")
    plt.tight_layout()
    out3b = os.path.join(PLOTS_DIR, "f1_by_arch_seq.png")
    plt.savefig(out3b)
    plt.close()
    print("Saved:", out3b)

    # --- Plot 4: Heatmap of mean F1 (Architecture x Seq Length) using imshow ---
    # prepare matrix rows=Architecture, cols=Seq Length
    archs = list(pivot_f1.columns)
    seqs = list(pivot_f1.index)
    mat = pivot_f1.values.T  # shape (n_arch, n_seq)
    plt.figure(figsize=(6,4))
    im = plt.imshow(mat, aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Mean F1')
    plt.yticks(range(len(archs)), archs)
    plt.xticks(range(len(seqs)), seqs)
    plt.xlabel("Sequence Length")
    plt.title("Heatmap: Mean F1 (Architecture × Seq Length)")
    out4 = os.path.join(PLOTS_DIR, "f1_heatmap_arch_seq.png")
    plt.tight_layout()
    plt.savefig(out4)
    plt.close()
    print("Saved:", out4)
else:
    print("Skipping architecture-based plots because 'Architecture' column not present in metrics.csv")

# --- Plot 5: Loss curves for top models if history pickles present ---
# history files saved as <modelname>_history.pkl in results/models/ by train.py
if os.path.isdir(MODELS_DIR):
    histories = [f for f in os.listdir(MODELS_DIR) if f.endswith("_history.pkl") or f.endswith("_history.pkl".lower())]
    if histories:
        # limit to top 4 histories by F1 to avoid too many plots
        top_models = df.sort_values("F1", ascending=False).head(4)["Model"].tolist()
        for model_name in top_models:
            hist_fn = os.path.join(MODELS_DIR, model_name + "_history.pkl")
            if not os.path.isfile(hist_fn):
                # try alternative naming
                candidates = [h for h in histories if h.startswith(model_name)]
                if candidates:
                    hist_fn = os.path.join(MODELS_DIR, candidates[0])
                else:
                    continue
            try:
                hist = pd.read_pickle(hist_fn)
                if "train_loss" in hist.columns and "val_loss" in hist.columns:
                    plt.figure()
                    plt.plot(hist["train_loss"], label="train_loss")
                    plt.plot(hist["val_loss"], label="val_loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Training Loss: {model_name}")
                    plt.legend()
                    out_hist = os.path.join(PLOTS_DIR, f"loss_{model_name}.png")
                    plt.tight_layout()
                    plt.savefig(out_hist)
                    plt.close()
                    print("Saved:", out_hist)
            except Exception as e:
                print("Could not load history for", model_name, "-", e)
    else:
        print("No history pickles found in", MODELS_DIR)
else:
    print("Models directory not found:", MODELS_DIR)

print("\nAll done. Inspect the plots in", PLOTS_DIR)
