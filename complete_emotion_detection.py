# complete_emotion_detection.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============ CONFIG ============
class TrainConfig:
    batch_size = 32
    lr = 1e-3
    epochs = 10
    hidden_dim = 128
    output_dir = "emotion_artifacts"
    os.makedirs(output_dir, exist_ok=True)

# ============ DATASET ============
class EmotionDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, scaler=None, fit_scaler=False):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[label_col].values.astype(np.int64)

        if scaler:
            if fit_scaler:
                self.X = scaler.fit_transform(self.X)
            else:
                self.X = scaler.transform(self.X)
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============ MODEL ============
class EmotionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ============ TRAIN & EVAL ============
def train_and_eval(model, train_loader, val_loader, cfg, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device).float(), yb.to(device).long()
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "emotion_model.pth"))

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device).float(), yb.to(device).long()
            out = model(Xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.save(os.path.join(cfg.output_dir, "confusion_matrix.npy"), cm)

    return model, cm, y_true, y_pred

# ============ DEMO ============
def demo_predictions(model, val_loader, idx2label):
    model.eval()
    X_batch, y_batch = next(iter(val_loader))
    with torch.no_grad():
        logits = model(X_batch.float())
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    # Show first sample
    pred_idx = preds[0]
    true_idx = y_batch[0].item()
    print("\n1. Predicted Emotion Label\n")
    print(f"Predicted Emotion: {idx2label[pred_idx]}")
    print(f"True Emotion     : {idx2label[true_idx]}")

    print("\n2. Probability Distribution Across Emotions\n")
    for lbl, p in zip(idx2label.values(), probs[0]):
        print(f"{lbl}: {p:.2f}")

def plot_confusion_matrix(cm, idx2label):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=idx2label.values(),
                yticklabels=idx2label.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ============ PIPELINE ============
def run_pipeline_from_globals(df):
    cfg = TrainConfig()

    label_col = "label"
    feature_cols = [c for c in df.columns if c != label_col]

    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    idx2label = {i: lbl for i, lbl in enumerate(le.classes_)}

    X_train, X_val = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    scaler = StandardScaler()

    train_ds = EmotionDataset(X_train, feature_cols, label_col, scaler, fit_scaler=True)
    val_ds = EmotionDataset(X_val, feature_cols, label_col, scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = EmotionNet(len(feature_cols), cfg.hidden_dim, num_classes=len(idx2label))
    model, cm, y_true, y_pred = train_and_eval(model, train_loader, val_loader, cfg, len(idx2label))

    # === DEMO OUTPUTS ===
    print("\n=== DEMO OUTPUTS ===")
    demo_predictions(model, val_loader, idx2label)

    print("\n=== CONFUSION MATRIX ===")
    plot_confusion_matrix(cm, idx2label)

    print("\n=== CLASSIFICATION METRICS ===")
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values())))

'''
import os
import math
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# -----------------------------
# Configuration and Utilities
# -----------------------------

EMOTIONS = ["Baseline", "Amusement", "Stress"]
LABEL_TO_ID = {name: i for i, name in enumerate(EMOTIONS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}
NUM_CLASSES = len(EMOTIONS)

WESAD_NUMERIC_TO_STR = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation"
}

@dataclass
class TrainConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 60
    weight_decay: float = 1e-4
    patience: int = 8  # early stopping patience
    num_workers: int = 0
    seed: int = 42
    out_dir: str = "emotion_artifacts"


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset wrapper
# -----------------------------

class EmotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, "X must be 2D [N, D]"
        assert y.ndim == 1, "y must be 1D [N]"
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Model
# -----------------------------

class EmotionNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Core pipeline
# -----------------------------

def prepare_data_from_final_df(final_features_df: pd.DataFrame,
                               config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if "label" not in final_features_df.columns:
        raise ValueError("final_features_df must contain a 'label' column with WESAD numeric labels.")

    # Select useful rows: map numeric -> string; keep only target emotions
    labels_numeric = final_features_df["label"].astype("float64")  # may contain NaN
    keep_mask = labels_numeric.isin([1, 2, 3])
    df_kept = final_features_df.loc[keep_mask].copy()

    # Drop rows with any NaN in features
    feature_cols = [c for c in df_kept.columns if c != "label"]
    df_kept = df_kept.dropna(subset=feature_cols + ["label"])

    # Map numeric labels -> strings -> IDs
    df_kept["label_str"] = df_kept["label"].astype(int).map(WESAD_NUMERIC_TO_STR)
    df_kept = df_kept[df_kept["label_str"].isin(EMOTIONS)]
    df_kept["label_id"] = df_kept["label_str"].map(LABEL_TO_ID)

    # Feature matrix
    X = df_kept[feature_cols].to_numpy()
    y = df_kept["label_id"].to_numpy()

    return X, y, feature_cols


def stratified_splits(X: np.ndarray, y: np.ndarray, config: TrainConfig):
    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    # Split train/val
    val_fraction_of_trainval = config.val_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction_of_trainval, stratify=y_trainval, random_state=config.seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    # Avoid div by zero
    class_counts[class_counts == 0] = 1.0
    weights = class_counts.sum() / class_counts
    return torch.tensor(weights, dtype=torch.float32)


def train_and_eval(final_features_df: pd.DataFrame,
                   config: Optional[TrainConfig] = None) -> Dict:
    """
    Expects final_features_df with numeric feature columns and a 'label' column with WESAD numeric labels.
    Returns logs and saves artifacts under config.out_dir.
    """
    if config is None:
        config = TrainConfig()
    set_seed(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    # Prepare data
    X, y, feature_cols = prepare_data_from_final_df(final_features_df, config)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splits
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(X_scaled, y, config)

    # Datasets & DataLoaders
    ds_train = EmotionDataset(X_train, y_train)
    ds_val   = EmotionDataset(X_val, y_val)
    ds_test  = EmotionDataset(X_test, y_test)

    train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader   = DataLoader(ds_val,   batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionNet(input_dim=X_train.shape[1], num_classes=NUM_CLASSES).to(device)

    # Loss with class weights
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Training loop with early stopping on val loss
    best_val_loss = float("inf")
    patience_left = config.patience
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / max(1, len(val_loader))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_left = config.patience
            torch.save(model.state_dict(), os.path.join(config.out_dir, "emotion_net_best.pt"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Load best and evaluate
    best_path = os.path.join(config.out_dir, "emotion_net_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    def predict_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        preds_all, y_all = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.append(preds)
                y_all.append(yb.numpy())
        return np.concatenate(preds_all), np.concatenate(y_all)

    y_pred_val, y_val_true = predict_loader(val_loader)
    y_pred_test, y_test_true = predict_loader(test_loader)

    # Metrics
    report_val = classification_report(y_val_true, y_pred_val, target_names=EMOTIONS, digits=3, output_dict=False)
    report_test = classification_report(y_test_true, y_pred_test, target_names=EMOTIONS, digits=3, output_dict=False)
    print("\nValidation classification report:\n", report_val)
    print("\nTest classification report:\n", report_test)

    cm = confusion_matrix(y_test_true, y_pred_test)

    # Save artifacts: scaler, label maps, feature columns, confusion matrix, reports
    import joblib
    joblib.dump(scaler, os.path.join(config.out_dir, "scaler.joblib"))
    with open(os.path.join(config.out_dir, "label_maps.json"), "w") as f:
        json.dump({"EMOTIONS": EMOTIONS, "LABEL_TO_ID": LABEL_TO_ID, "ID_TO_LABEL": ID_TO_LABEL}, f, indent=2)
    with open(os.path.join(config.out_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    np.save(os.path.join(config.out_dir, "confusion_matrix.npy"), cm)

    # Optionally plot and save confusion matrix using matplotlib (no seaborn)
    try:
        import matplotlib.pyplot as plt
        import itertools

        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix (Test)")
        plt.colorbar()
        tick_marks = np.arange(len(EMOTIONS))
        plt.xticks(tick_marks, EMOTIONS, rotation=45, ha="right")
        plt.yticks(tick_marks, EMOTIONS)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(config.out_dir, "confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print("Plotting CM failed:", e)

    return {
        "val_report": report_val,
        "test_report": report_test,
        "confusion_matrix": cm,
        "artifacts_dir": config.out_dir,
        "n_features": X.shape[1]
    }


# -----------------------------
# Convenience: Run if used in a notebook
# -----------------------------

def run_pipeline_from_globals(globals_dict: dict, out_dir: str = "emotion_artifacts"):
    """
    Call this in a notebook cell **after** you have `final_features_df` defined.
    Example:
        from complete_emotion_detection import run_pipeline_from_globals
        results = run_pipeline_from_globals(globals())
    """
    if "final_features_df" not in globals_dict:
        raise RuntimeError("final_features_df not found in the current notebook globals. "
                           "Run feature engineering cells first.")
    df = globals_dict["final_features_df"]
    cfg = TrainConfig(out_dir=out_dir)
    return train_and_eval(df, cfg)
'''