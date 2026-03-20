"""
Test script for evaluating the trained emotion classifier.
Computes F1, Precision, Recall metrics on a held-out test set.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- TEXT PREPROCESSING (same as training) ---
def preprocess_text(text):
    text = str(text).lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\u1000-\u109F\w\s]', '', text)
    return text

# --- ATTENTION MODULE ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1), weights

# --- MODEL (auto-detects architecture from checkpoint) ---
class EmotionalBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate=0.3, use_layer_norm=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        context, weights = self.attention(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# --- EVALUATOR CLASS ---
class ModelEvaluator:
    def __init__(self, model_path="./models/eliza_mm.pth"):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.word2id = {}
        self.id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        self.model = None
        self.max_len = 100

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.word2id = checkpoint['vocab']
        state_dict = checkpoint['state']

        # Auto-detect architecture from checkpoint
        embed_dim = state_dict['embedding.weight'].shape[1]
        hidden_dim = state_dict['fc.weight'].shape[1] // 2  # fc input is hidden_dim * 2
        use_layer_norm = 'layer_norm.weight' in state_dict

        print(f"[*] Detected architecture: embed_dim={embed_dim}, hidden_dim={hidden_dim}, layer_norm={use_layer_norm}")

        self.model = EmotionalBiLSTM(
            len(self.word2id), embed_dim=embed_dim, hidden_dim=hidden_dim,
            output_dim=6, dropout_rate=0.3, use_layer_norm=use_layer_norm
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[*] Model loaded from {self.model_path}")
        print(f"[*] Vocabulary size: {len(self.word2id)}")
        print(f"[*] Device: {self.device}")

    def tokenize(self, text):
        clean_text = preprocess_text(text)
        tokens = [self.word2id.get(w, 1) for w in clean_text.split()][:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

    def predict(self, text):
        tokens = self.tokenize(text)
        with torch.no_grad():
            output = self.model(torch.tensor([tokens]).to(self.device))
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs).item()
            return idx, probs[0][idx].item()

    def predict_batch(self, texts):
        all_tokens = [self.tokenize(t) for t in texts]
        batch = torch.tensor(all_tokens).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()
        return preds, confidences

    def evaluate(self, test_path, batch_size=64):
        print(f"\n[*] Loading test data from {test_path}")
        df = pd.read_csv(test_path)

        # Detect label column
        label_col = 'label' if 'label' in df.columns else 'emotions'

        # Drop NaN
        before = len(df)
        df = df.dropna(subset=[label_col, 'text'])
        dropped = before - len(df)
        if dropped:
            print(f"[!] Dropped {dropped} rows with NaN values")

        texts = df['text'].tolist()
        raw_labels = df[label_col].tolist()

        # Convert labels to integers
        if isinstance(raw_labels[0], str):
            unique_labels = sorted(list(set(raw_labels)))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            true_labels = [label_to_id[l] for l in raw_labels]
            self.id2label = {idx: label for label, idx in label_to_id.items()}
            print(f"[*] String labels detected: {label_to_id}")
        else:
            true_labels = [int(l) for l in raw_labels]

        print(f"[*] Test samples: {len(texts)}")

        # Batch prediction
        all_preds = []
        all_confs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            preds, confs = self.predict_batch(batch_texts)
            all_preds.extend(preds)
            all_confs.extend(confs)

        all_preds = np.array(all_preds)
        true_labels = np.array(true_labels)

        # Metrics
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)

        target_names = [self.id2label.get(i, f"Class_{i}") for i in range(6)]
        report = classification_report(true_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
        print(report)

        # Overall metrics
        accuracy = (all_preds == true_labels).mean()
        f1_macro = f1_score(true_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, all_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(true_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, all_preds, average='macro', zero_division=0)

        print("="*60)
        print("SUMMARY METRICS")
        print("="*60)
        print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1 (macro):        {f1_macro:.4f}")
        print(f"F1 (weighted):     {f1_weighted:.4f}")
        print(f"Precision (macro): {precision_macro:.4f}")
        print(f"Recall (macro):    {recall_macro:.4f}")
        print(f"Avg Confidence:    {np.mean(all_confs):.4f}")
        print("="*60)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, all_preds)
        print("\nCONFUSION MATRIX:")
        print(cm)

        # Save confusion matrix plot
        self._plot_confusion_matrix(cm, target_names)

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm
        }

    def _plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('./assets/confusion_matrix.png', dpi=150)
        print("\n[*] Confusion matrix saved to confusion_matrix.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained emotion classifier")
    parser.add_argument("--model", default="./models/eliza_mm.pth", help="Path to trained model")
    parser.add_argument("--test_data", default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()

    evaluator = ModelEvaluator(model_path=args.model)
    evaluator.load_model()
    results = evaluator.evaluate(args.test_data, batch_size=args.batch_size)

    print("\n[*] Evaluation complete!")


if __name__ == "__main__":
    main()
