"""
ABSA Multi-Task Training — MULTI-SEED VERSION
================================================

✅ RoBERTa encoder (robust, no torch issues)
✅ Multi-task ABSA model
✅ Single-task RoBERTa baseline
✅ Random Forest & Gradient Boosting
✅ Optimized loss weighting
✅ RTX 4090 GPU optimized
✅ Multi-seed evaluation for statistical validity
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    roc_curve, auc, confusion_matrix,
)
from scipy.sparse import hstack, csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    # Data
    CSV_FILE = r"C:\Users\Administrator\Desktop\amazon_reviews.csv"
    MAX_ROWS = 100000
    TEST_SIZE = 0.20
    VAL_SIZE = 0.10

    # Training
    EPOCHS = 5
    BATCH_SIZE = 128
    MAX_LENGTH = 256
    LEARNING_RATE = 1.5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.15

    # Model
    ENCODER = "roberta-base"
    DROPOUT = 0.4
    HIDDEN_DIM = 256

    # Loss weighting (optimized)
    SENTIMENT_LOSS_WEIGHT = 0.5
    RECOMMEND_LOSS_WEIGHT = 1.0

    # Output
    OUTPUT_DIR = "results"

    # ✅ MULTI-SEED: list of seeds to run
    SEEDS = [42, 123, 456, 789, 1024]

    # For backward compatibility — used as default when not looping
    RANDOM_SEED = 42

    # Flags
    RUN_BASELINES = True       # only runs once (deterministic with TF-IDF)
    RUN_RF_GB = True           # only runs once
    RUN_SINGLE_TASK = True     # runs per seed
    RUN_MULTI_TASK = True      # runs per seed
    SAVE_PLOTS = True


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════════
#  SEED UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slight speed cost, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE SETUP
# ═══════════════════════════════════════════════════════════════════════════

class DeviceManager:
    @staticmethod
    def setup():
        if not torch.cuda.is_available():
            print("❌ CUDA NOT AVAILABLE")
            sys.exit(1)

        device = torch.device("cuda:0")

        props = torch.cuda.get_device_properties(0)
        print("\n" + "=" * 70)
        print("GPU SETUP")
        print("=" * 70)
        print(f"🎮 Device: {props.name}")
        print(f"📦 VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"🔥 Torch: {torch.__version__}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled")
        print(f"✅ Using: {cfg.ENCODER}")
        print(f"✅ Seeds to evaluate: {cfg.SEEDS}\n")

        return device


device = DeviceManager.setup()


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader_Custom:
    RENAME_MAP = {
        "reviewtext": "review_text", "review_text": "review_text", "text": "review_text",
        "body": "review_text", "overall": "rating", "rating": "rating", "stars": "rating",
        "summary": "summary", "verified": "verified_purchase", "verified_purchase": "verified_purchase",
        "vote": "helpful_votes", "helpful_votes": "helpful_votes", "helpful_vote": "helpful_votes",
        "asin": "product_id", "parent_asin": "product_id", "reviewerid": "reviewer_id",
        "review text": "review_text", "review title": "summary",
    }

    @staticmethod
    def load(path, max_rows=None, seed=42):
        print("=" * 70)
        print("STEP 1: LOAD DATA")
        print("=" * 70)

        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            sys.exit(1)

        print(f"Loading {path}...")
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        print(f"✓ Loaded {len(df):,} rows")

        df.columns = [c.strip().lower() for c in df.columns]
        renames = {c: DataLoader_Custom.RENAME_MAP[c] for c in df.columns if c in DataLoader_Custom.RENAME_MAP}
        df = df.rename(columns=renames)

        if "review_text" not in df.columns:
            print(f"❌ No 'review_text' column found")
            sys.exit(1)

        if "summary" in df.columns:
            def combine(row):
                s = str(row["summary"]).strip()
                t = str(row["review_text"]).strip()
                if s and s.lower() not in ("nan", "none", ""):
                    return s + ". " + t
                return t

            df["review_text"] = df.apply(combine, axis=1)

        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(
                df["rating"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )

        if "rating" not in df.columns:
            print("❌ No 'rating' column found")
            sys.exit(1)

        df["label"] = (df["rating"] >= 4).astype(int)

        df = df.dropna(subset=["review_text"]).copy()
        df = df.drop_duplicates(subset=["review_text"]).copy()
        df["review_text"] = df["review_text"].astype(str).str.strip()
        df = df[df["review_text"].str.len() > 10].reset_index(drop=True)

        if max_rows and len(df) > max_rows:
            print(f"✓ Sampling {max_rows:,} rows (seed={seed})")
            df = df.sample(max_rows, random_state=seed).reset_index(drop=True)

        pos_rate = df["label"].mean()
        print(f"✓ Final rows: {len(df):,}")
        print(f"✓ Positive: {pos_rate:.1%} | Negative: {1 - pos_rate:.1%}\n")

        return df


# ═══════════════════════════════════════════════════════════════════════════
#  ASPECT FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class AspectExtractor:
    LEXICON = {
        "quality": ["quality", "build", "material", "defect", "broken", "damage", "excellent"],
        "price": ["price", "cost", "value", "worth", "expensive", "cheap", "deal"],
        "shipping": ["shipping", "delivery", "arrived", "packaging", "late", "fast"],
        "durability": ["durable", "durability", "last", "sturdy", "wear", "tear"],
        "design": ["design", "look", "style", "color", "size", "fit", "aesthetic"],
        "service": ["service", "support", "customer", "refund", "return", "helpful"],
    }

    @staticmethod
    def extract(texts):
        rows = []
        for text in texts:
            tl = str(text).lower()
            row = {
                f"asp_{asp}": sum(1 for kw in kws if kw in tl) / max(len(kws), 1)
                for asp, kws in AspectExtractor.LEXICON.items()
            }
            rows.append(row)
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════════════

class BaselineModels:
    @staticmethod
    def train_and_evaluate(df_train, df_test, include_rf_gb=False):
        print("=" * 70)
        print("STEP 2: BASELINES (CPU)")
        print("=" * 70)

        train_texts = df_train["review_text"].fillna("").astype(str).tolist()
        test_texts = df_test["review_text"].fillna("").astype(str).tolist()
        y_train = df_train["label"].values
        y_test = df_test["label"].values

        print("Computing TF-IDF...", end=" ")
        tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
        )
        X_tr = tfidf.fit_transform(train_texts)
        X_te = tfidf.transform(test_texts)
        print(f"✓ ({X_tr.shape[1]:,} features)")

        print("Extracting aspects...", end=" ")
        asp_tr = AspectExtractor.extract(train_texts)
        asp_te = AspectExtractor.extract(test_texts)
        X_tr = hstack([X_tr, csr_matrix(asp_tr.values)])
        X_te = hstack([X_te, csr_matrix(asp_te.values)])
        print("✓\n")

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced",
                C=1.0, random_state=cfg.RANDOM_SEED, n_jobs=-1,
            ),
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "SVM": CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", max_iter=2000,
                          random_state=cfg.RANDOM_SEED, dual=False)
            ),
        }

        if include_rf_gb:
            print("⏳ Training tree-based models (2-3 minutes)...\n")
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced",
                n_jobs=-1, random_state=cfg.RANDOM_SEED, verbose=0,
            )
            models["Gradient Boosting"] = GradientBoostingClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                subsample=0.8, random_state=cfg.RANDOM_SEED, verbose=0,
            )

        results = []
        roc_curves = {}

        print("Training models:")
        for name, model in models.items():
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                if "Naive Bayes" in name:
                    X_tr_nb = X_tr.copy()
                    X_tr_nb.data = np.abs(X_tr_nb.data)
                    X_te_nb = X_te.copy()
                    X_te_nb.data = np.abs(X_te_nb.data)
                    model.fit(X_tr_nb, y_train)
                    y_pred = model.predict(X_te_nb)
                    proba = model.predict_proba(X_te_nb)[:, 1]
                else:
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_te)
                    proba = model.predict_proba(X_te)[:, 1]

                elapsed = time.time() - t0

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                fpr, tpr, _ = roc_curve(y_test, proba)
                auc_score = auc(fpr, tpr)
                roc_curves[name] = (fpr, tpr)

                results.append({
                    "Model": name,
                    "Type": "Classical",
                    "Accuracy": round(acc, 4),
                    "F1": round(f1, 4),
                    "F1(+)": round(cr.get("1", {}).get("f1-score", 0), 4),
                    "F1(-)": round(cr.get("0", {}).get("f1-score", 0), 4),
                    "AUC": round(auc_score, 4),
                })

                print(f"F1={f1:.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")

        print()
        return results, roc_curves


# ═══════════════════════════════════════════════════════════════════════════
#  GPU TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════

class GPUTokenizer:
    @staticmethod
    def tokenize_batch(texts, tokenizer, max_length, device, batch_size=512):
        input_ids_list = []
        attention_masks = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", leave=True):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids_list.append(encoded["input_ids"].to(device))
            attention_masks.append(encoded["attention_mask"].to(device))

        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)

        print(f"✓ Tokenized: {input_ids.shape}\n")
        return input_ids, attention_mask


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, sentiment_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
            "sentiment_label": self.sentiment_labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SingleTaskRoBERTa(nn.Module):
    """Single-task RoBERTa: recommendation only"""

    def __init__(self, encoder_name, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.dense = nn.Linear(hidden, cfg.HIDDEN_DIM)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.activation = nn.GELU()
        self.recommend_head = nn.Linear(cfg.HIDDEN_DIM, 2)

        self.recommend_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]

        hidden = self.activation(self.dense(cls))
        hidden = self.dropout(hidden)

        logits = self.recommend_head(hidden)

        loss = None
        if labels is not None:
            loss = self.recommend_loss(logits, labels)

        return loss, logits


class MultiTaskRoBERTa(nn.Module):
    """Multi-task RoBERTa with optimized loss weighting"""

    def __init__(self, encoder_name, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.shared_dense = nn.Linear(hidden, cfg.HIDDEN_DIM)
        self.shared_dropout = nn.Dropout(cfg.DROPOUT)
        self.activation = nn.GELU()

        self.sentiment_dense = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM // 2)
        self.sentiment_dropout = nn.Dropout(cfg.DROPOUT)
        self.sentiment_head = nn.Linear(cfg.HIDDEN_DIM // 2, 3)

        self.recommend_dense = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM // 2)
        self.recommend_dropout = nn.Dropout(cfg.DROPOUT)
        self.recommend_head = nn.Linear(cfg.HIDDEN_DIM // 2, 2)

        self.sentiment_loss = nn.CrossEntropyLoss()
        self.recommend_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, sentiment_labels=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]

        shared = self.activation(self.shared_dense(cls))
        shared = self.shared_dropout(shared)

        sentiment_hidden = self.activation(self.sentiment_dense(shared))
        sentiment_hidden = self.sentiment_dropout(sentiment_hidden)
        sentiment_logits = self.sentiment_head(sentiment_hidden)

        recommend_hidden = self.activation(self.recommend_dense(shared))
        recommend_hidden = self.recommend_dropout(recommend_hidden)
        recommend_logits = self.recommend_head(recommend_hidden)

        loss = None
        if sentiment_labels is not None and labels is not None:
            sent_loss = cfg.SENTIMENT_LOSS_WEIGHT * self.sentiment_loss(sentiment_logits, sentiment_labels)
            rec_loss = cfg.RECOMMEND_LOSS_WEIGHT * self.recommend_loss(recommend_logits, labels)
            loss = sent_loss + rec_loss

        return loss, recommend_logits, sentiment_logits


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    @staticmethod
    def rating_to_sentiment(rating):
        r = float(rating) if not np.isnan(float(rating)) else 3.0
        if r <= 2:
            return 0
        elif r == 3:
            return 1
        else:
            return 2

    @staticmethod
    def train_single_task(train_ids, train_mask, train_labels,
                          val_ids, val_mask, val_labels,
                          device, output_dir, seed):
        print("=" * 70)
        print(f"SINGLE-TASK RoBERTa TRAINING (seed={seed})")
        print("=" * 70 + "\n")

        train_ds = TextDataset(train_ids, train_mask, train_labels, [0] * len(train_labels))
        val_ds = TextDataset(val_ids, val_mask, val_labels, [0] * len(val_labels))

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = torch.tensor(
            len(train_labels) / (len(unique) * counts),
            dtype=torch.float32,
        ).to(device)

        model = SingleTaskRoBERTa(cfg.ENCODER, class_weights=class_weights).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = len(train_loader) * cfg.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

        best_f1 = 0.0
        best_model_path = Path(output_dir) / f"best_single_task_seed{seed}.pt"
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            train_losses = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", ncols=80):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    loss, _ = model(ids, mask, labels=labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            model.eval()
            val_preds = []
            val_labels_list = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", ncols=80):
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                        _, logits = model(ids, mask)

                    preds = logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_list.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)

            print(f"\nEpoch {epoch} | Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Saved best model")

        print(f"\nBest Val F1: {best_f1:.4f}")
        return model, best_model_path

    @staticmethod
    def train_multi_task(train_ids, train_mask, train_labels, train_sentiments,
                         val_ids, val_mask, val_labels, val_sentiments,
                         device, output_dir, seed):
        print("=" * 70)
        print(f"MULTI-TASK RoBERTa TRAINING (seed={seed})")
        print("=" * 70)
        print(f"Loss Weighting: Sentiment={cfg.SENTIMENT_LOSS_WEIGHT}, Recommend={cfg.RECOMMEND_LOSS_WEIGHT}\n")

        train_ds = TextDataset(train_ids, train_mask, train_labels, train_sentiments)
        val_ds = TextDataset(val_ids, val_mask, val_labels, val_sentiments)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = torch.tensor(
            len(train_labels) / (len(unique) * counts),
            dtype=torch.float32,
        ).to(device)

        model = MultiTaskRoBERTa(cfg.ENCODER, class_weights=class_weights).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = len(train_loader) * cfg.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

        best_f1 = 0.0
        best_model_path = Path(output_dir) / f"best_multi_task_seed{seed}.pt"
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            train_losses = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", ncols=80):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                sentiments = batch["sentiment_label"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    loss, _, _ = model(ids, mask, sentiment_labels=sentiments, labels=labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            model.eval()
            val_preds = []
            val_labels_list = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", ncols=80):
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    sentiments = batch["sentiment_label"].to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                        _, logits, _ = model(ids, mask, sentiment_labels=sentiments, labels=labels)

                    preds = logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_list.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)

            print(f"\nEpoch {epoch} | Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Saved best model")

        print(f"\nBest Val F1: {best_f1:.4f}")
        return model, best_model_path

    @staticmethod
    def evaluate_single_task(model, test_ids, test_mask, test_labels, device):
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

        test_ds = TextDataset(test_ids, test_mask, test_labels, [0] * len(test_labels))
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=0)

        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating ST"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    _, logits = model(ids, mask)

                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    @staticmethod
    def evaluate_multi_task(model, test_ids, test_mask, test_labels, device):
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

        test_ds = TextDataset(test_ids, test_mask, test_labels, [0] * len(test_labels))
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=0)

        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating MT"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    _, logits, _ = model(ids, mask)

                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

class Visualizer:
    @staticmethod
    def plot_results(all_results, roc_curves, output_dir):
        print("\nGenerating plots...")
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_results).sort_values("F1", ascending=False)

        # Plot 1: F1 Comparison
        fig, ax = plt.subplots(figsize=(12, 7))
        models = df["Model"]
        f1s = df["F1"].astype(float)

        colors = []
        for model in df["Model"]:
            if "Multi-Task" in model:
                colors.append("#1D9E75")
            elif "Single-Task" in model:
                colors.append("#534AB7")
            elif "Random Forest" in model or "Gradient Boosting" in model:
                colors.append("#FF9500")
            else:
                colors.append("#888780")

        bars = ax.barh(models, f1s, color=colors, height=0.6, edgecolor="black", linewidth=1.5)
        ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=11, fontweight="bold")
        ax.set_xlabel("F1 Score", fontsize=12, fontweight="bold")
        ax.set_xlim(0.8, 0.96)
        ax.set_title("Model Comparison: RoBERTa vs Tree-Based vs Classical",
                     fontsize=13, fontweight="bold", pad=15)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1D9E75', edgecolor='black', label='Multi-Task RoBERTa'),
            Patch(facecolor='#534AB7', edgecolor='black', label='Single-Task RoBERTa'),
            Patch(facecolor='#FF9500', edgecolor='black', label='Tree-Based Models'),
            Patch(facecolor='#888780', edgecolor='black', label='Classical Baselines'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()
        plt.savefig(plots_dir / "f1_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: ROC Curves
        if roc_curves:
            fig, ax = plt.subplots(figsize=(9, 8))

            colors_roc = {
                "Multi-Task RoBERTa": "#1D9E75",
                "Single-Task RoBERTa": "#534AB7",
                "Random Forest": "#FF9500",
                "Gradient Boosting": "#FFB84D",
                "Logistic Regression": "#D85A30",
                "SVM": "#378ADD",
                "Naive Bayes": "#BA7517",
            }

            for name, (fpr, tpr) in roc_curves.items():
                auc_score = auc(fpr, tpr)
                color = colors_roc.get(name, "#888780")
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})",
                        color=color, lw=2.5)

            ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
            ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(plots_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Plot 3: Metrics Heatmap
        fig, ax = plt.subplots(figsize=(11, 7))
        metrics_df = df[["Model", "Accuracy", "F1", "AUC"]].set_index("Model")

        sns.heatmap(metrics_df.T, annot=True, fmt=".4f", cmap="RdYlGn",
                    cbar_kws={"label": "Score"}, ax=ax, linewidths=1, linecolor="black")
        ax.set_title("Performance Metrics Heatmap", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ All plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — MULTI-SEED LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("ABSA MULTI-SEED EVALUATION")
    print(f"Seeds: {cfg.SEEDS}")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    output_dir = Path(cfg.OUTPUT_DIR)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load data ONCE (use first seed for sampling) ──
    # Data loading and sampling is done once. The seed variation affects
    # model weight initialization and DataLoader shuffle order, NOT the
    # data split — this isolates training variance from data variance.
    set_seed(cfg.SEEDS[0])
    df = DataLoader_Custom.load(cfg.CSV_FILE, max_rows=cfg.MAX_ROWS, seed=cfg.SEEDS[0])

    # ── STEP 2: Split ONCE (same split for all seeds) ──
    print("=" * 70)
    print("SPLITTING DATA (fixed across all seeds)")
    print("=" * 70)

    df_trainval, df_test = train_test_split(
        df, test_size=cfg.TEST_SIZE,
        random_state=cfg.SEEDS[0],  # fixed split
        stratify=df["label"],
    )

    val_size = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size,
        random_state=cfg.SEEDS[0],  # fixed split
        stratify=df_trainval["label"],
    )

    print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}\n")

    # ── STEP 3: Baselines ONCE (deterministic) ──
    baseline_results = []
    roc_curves = {}

    if cfg.RUN_BASELINES:
        baseline_results, roc_curves = BaselineModels.train_and_evaluate(
            df_train, df_test, include_rf_gb=cfg.RUN_RF_GB
        )

    # ── STEP 4: Tokenize ONCE (deterministic) ──
    print("=" * 70)
    print("TOKENIZING (one-time, shared across all seeds)")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.ENCODER, use_fast=True)

    train_ids, train_mask = GPUTokenizer.tokenize_batch(
        df_train["review_text"].astype(str).tolist(),
        tokenizer, cfg.MAX_LENGTH, device
    )
    val_ids, val_mask = GPUTokenizer.tokenize_batch(
        df_val["review_text"].astype(str).tolist(),
        tokenizer, cfg.MAX_LENGTH, device
    )
    test_ids, test_mask = GPUTokenizer.tokenize_batch(
        df_test["review_text"].astype(str).tolist(),
        tokenizer, cfg.MAX_LENGTH, device
    )

    train_sentiments = [Trainer.rating_to_sentiment(r) for r in df_train["rating"].fillna(3)]
    val_sentiments = [Trainer.rating_to_sentiment(r) for r in df_val["rating"].fillna(3)]

    # ── STEP 5: Multi-seed training loop ──
    # Collectors for per-seed results
    seed_results_st = []   # Single-Task per seed
    seed_results_mt = []   # Multi-Task per seed

    for seed_idx, seed in enumerate(cfg.SEEDS):
        print("\n" + "█" * 70)
        print(f"  SEED {seed_idx + 1}/{len(cfg.SEEDS)}: seed={seed}")
        print("█" * 70 + "\n")

        # Set all random states for this seed
        set_seed(seed)

        # Clear GPU cache between runs
        torch.cuda.empty_cache()

        # ── Single-Task ──
        if cfg.RUN_SINGLE_TASK:
            model_st, best_path_st = Trainer.train_single_task(
                train_ids, train_mask, df_train["label"].tolist(),
                val_ids, val_mask, df_val["label"].tolist(),
                device, str(output_dir / "models"), seed
            )

            model_st.load_state_dict(torch.load(best_path_st, map_location=device, weights_only=False))
            y_true_st, y_pred_st, y_prob_st = Trainer.evaluate_single_task(
                model_st, test_ids, test_mask, df_test["label"].tolist(), device
            )

            acc_st = accuracy_score(y_true_st, y_pred_st)
            f1_st = f1_score(y_true_st, y_pred_st, average="macro", zero_division=0)
            cr_st = classification_report(y_true_st, y_pred_st, output_dict=True, zero_division=0)
            fpr_st, tpr_st, _ = roc_curve(y_true_st, y_prob_st)
            auc_st = auc(fpr_st, tpr_st)

            seed_results_st.append({
                "Seed": seed,
                "Accuracy": acc_st,
                "Macro_F1": f1_st,
                "F1_pos": cr_st.get("1", {}).get("f1-score", 0),
                "F1_neg": cr_st.get("0", {}).get("f1-score", 0),
                "AUC": auc_st,
            })

            print(f"\n✓ ST seed={seed}: F1={f1_st:.4f}, AUC={auc_st:.4f}")

            # Save ROC for first seed only (for plots)
            if seed_idx == 0:
                roc_curves["Single-Task RoBERTa"] = (fpr_st, tpr_st)

            # Free model memory
            del model_st
            torch.cuda.empty_cache()

        # ── Multi-Task ──
        if cfg.RUN_MULTI_TASK:
            # Re-set seed before MT training for this seed
            set_seed(seed)

            model_mt, best_path_mt = Trainer.train_multi_task(
                train_ids, train_mask, df_train["label"].tolist(), train_sentiments,
                val_ids, val_mask, df_val["label"].tolist(), val_sentiments,
                device, str(output_dir / "models"), seed
            )

            model_mt.load_state_dict(torch.load(best_path_mt, map_location=device, weights_only=False))
            y_true_mt, y_pred_mt, y_prob_mt = Trainer.evaluate_multi_task(
                model_mt, test_ids, test_mask, df_test["label"].tolist(), device
            )

            acc_mt = accuracy_score(y_true_mt, y_pred_mt)
            f1_mt = f1_score(y_true_mt, y_pred_mt, average="macro", zero_division=0)
            cr_mt = classification_report(y_true_mt, y_pred_mt, output_dict=True, zero_division=0)
            fpr_mt, tpr_mt, _ = roc_curve(y_true_mt, y_prob_mt)
            auc_mt = auc(fpr_mt, tpr_mt)

            seed_results_mt.append({
                "Seed": seed,
                "Accuracy": acc_mt,
                "Macro_F1": f1_mt,
                "F1_pos": cr_mt.get("1", {}).get("f1-score", 0),
                "F1_neg": cr_mt.get("0", {}).get("f1-score", 0),
                "AUC": auc_mt,
            })

            print(f"\n✓ MT seed={seed}: F1={f1_mt:.4f}, AUC={auc_mt:.4f}")

            if seed_idx == 0:
                roc_curves["Multi-Task RoBERTa"] = (fpr_mt, tpr_mt)

            del model_mt
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  MULTI-SEED SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 70)

    # ── Per-seed detail table ──
    if seed_results_st:
        df_st = pd.DataFrame(seed_results_st)
        print("\n── Single-Task RoBERTa (per seed) ──")
        print(df_st.to_string(index=False, float_format="%.4f"))

    if seed_results_mt:
        df_mt = pd.DataFrame(seed_results_mt)
        print("\n── Multi-Task RoBERTa (per seed) ──")
        print(df_mt.to_string(index=False, float_format="%.4f"))

    # ── Aggregated summary (THIS GOES IN YOUR THESIS) ──
    print("\n" + "=" * 70)
    print("THESIS-READY SUMMARY TABLE")
    print("=" * 70)
    print(f"(Report these numbers in Tables 4.2 and 4.3)\n")

    summary_rows = []

    if seed_results_st:
        df_st = pd.DataFrame(seed_results_st)
        summary_rows.append({
            "Model": "Single-Task RoBERTa",
            "Accuracy": f"{df_st['Accuracy'].mean():.4f} ± {df_st['Accuracy'].std():.4f}",
            "Macro_F1": f"{df_st['Macro_F1'].mean():.4f} ± {df_st['Macro_F1'].std():.4f}",
            "AUC": f"{df_st['AUC'].mean():.4f} ± {df_st['AUC'].std():.4f}",
            "F1_pos": f"{df_st['F1_pos'].mean():.4f} ± {df_st['F1_pos'].std():.4f}",
            "F1_neg": f"{df_st['F1_neg'].mean():.4f} ± {df_st['F1_neg'].std():.4f}",
        })

    if seed_results_mt:
        df_mt = pd.DataFrame(seed_results_mt)
        summary_rows.append({
            "Model": "Multi-Task RoBERTa",
            "Accuracy": f"{df_mt['Accuracy'].mean():.4f} ± {df_mt['Accuracy'].std():.4f}",
            "Macro_F1": f"{df_mt['Macro_F1'].mean():.4f} ± {df_mt['Macro_F1'].std():.4f}",
            "AUC": f"{df_mt['AUC'].mean():.4f} ± {df_mt['AUC'].std():.4f}",
            "F1_pos": f"{df_mt['F1_pos'].mean():.4f} ± {df_mt['F1_pos'].std():.4f}",
            "F1_neg": f"{df_mt['F1_neg'].mean():.4f} ± {df_mt['F1_neg'].std():.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── MTL gain analysis ──
    if seed_results_st and seed_results_mt:
        df_st = pd.DataFrame(seed_results_st)
        df_mt = pd.DataFrame(seed_results_mt)

        gains = df_mt["Macro_F1"].values - df_st["Macro_F1"].values
        print(f"\n── MTL Gain per seed ──")
        for i, seed in enumerate(cfg.SEEDS):
            print(f"  Seed {seed}: {gains[i]:+.4f} ({gains[i]*100:+.2f}%)")

        print(f"\n  Mean MTL gain: {gains.mean():+.4f} ({gains.mean()*100:+.2f}%)")
        print(f"  Std MTL gain:  {gains.std():.4f}")
        print(f"  Min gain:      {gains.min():+.4f}")
        print(f"  Max gain:      {gains.max():+.4f}")

        # Simple t-test significance
        if len(gains) >= 3:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(gains, 0)
            print(f"\n  One-sample t-test (H0: gain = 0):")
            print(f"    t = {t_stat:.4f}, p = {p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✓ Statistically significant at p < 0.05")
            else:
                print(f"    ✗ NOT significant at p < 0.05 (reframe as regularisation benefit)")

    # ── Save everything ──
    all_seed_results = {
        "single_task_per_seed": seed_results_st,
        "multi_task_per_seed": seed_results_mt,
        "summary": summary_rows,
        "seeds": cfg.SEEDS,
        "config": {
            "encoder": cfg.ENCODER,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.LEARNING_RATE,
            "sentiment_weight": cfg.SENTIMENT_LOSS_WEIGHT,
            "recommend_weight": cfg.RECOMMEND_LOSS_WEIGHT,
        }
    }

    with open(output_dir / "multi_seed_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2, default=str)
    print(f"\n✓ Saved: {output_dir / 'multi_seed_results.json'}")

    # Save CSV versions for easy copy-paste into thesis
    if seed_results_st:
        pd.DataFrame(seed_results_st).to_csv(output_dir / "single_task_per_seed.csv", index=False)
    if seed_results_mt:
        pd.DataFrame(seed_results_mt).to_csv(output_dir / "multi_task_per_seed.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "thesis_summary_table.csv", index=False)

    print(f"✓ Saved: thesis_summary_table.csv")

    # ── Generate plots (using first seed + baselines) ──
    if cfg.SAVE_PLOTS and baseline_results:
        # Build combined results using mean values for the plot
        plot_results = list(baseline_results)
        if seed_results_st:
            df_st = pd.DataFrame(seed_results_st)
            plot_results.append({
                "Model": "Single-Task RoBERTa",
                "Type": "Single-Task",
                "Accuracy": round(df_st["Accuracy"].mean(), 4),
                "F1": round(df_st["Macro_F1"].mean(), 4),
                "F1(+)": round(df_st["F1_pos"].mean(), 4),
                "F1(-)": round(df_st["F1_neg"].mean(), 4),
                "AUC": round(df_st["AUC"].mean(), 4),
            })
        if seed_results_mt:
            df_mt = pd.DataFrame(seed_results_mt)
            plot_results.append({
                "Model": "Multi-Task RoBERTa",
                "Type": "Multi-Task",
                "Accuracy": round(df_mt["Accuracy"].mean(), 4),
                "F1": round(df_mt["Macro_F1"].mean(), 4),
                "F1(+)": round(df_mt["F1_pos"].mean(), 4),
                "F1(-)": round(df_mt["F1_neg"].mean(), 4),
                "AUC": round(df_mt["AUC"].mean(), 4),
            })

        Visualizer.plot_results(plot_results, roc_curves, str(output_dir))

    print("\n" + "=" * 70)
    print(f"COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total seeds evaluated: {len(cfg.SEEDS)}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
