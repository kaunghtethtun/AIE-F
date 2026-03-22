## Hybrid-ELIZA (EN + Myanmar, Multi-Tokenizer)
## Demo code of applying rules + BiLSTM for AI Engineering Class (Fundamental)
## Written by Ye, Language Understanding Lab., Myanmar
## Last updated: 20 Mar 2026
## Reference code: https://www.kaggle.com/code/wjburns/eliza

import argparse
import os
import random
import re
import subprocess
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.metrics import classification_report, confusion_matrix  # ADDED: for validation/test report and confusion matrix


# --- 1. GLOBAL SCRIPT DATA ---
DEFAULT_OPPAWORD_SCRIPT = "/home/phantom/Desktop/Git/oppaWord/oppa_word.py"
DEFAULT_OPPAWORD_DICT = "/home/phantom/Desktop/Git/oppaWord/data/myg2p_mypos.dict"

# Added 'Rank' (3rd element in list). Higher = Higher Priority.
SCRIPTS = {
    "en": {
        "initials": ["How do you do. Please tell me your problem.", "Is something troubling you?"],
        "finals": ["Goodbye. It was nice talking to you.", "Your terminal will self-destruct in 5s."],
        "quits": ["bye", "quit", "exit"],
        "pres": {"don't": "dont", "i'm": "i am", "recollect": "remember", "machine": "computer"},
        "posts": {"am": "are", "i": "you", "my": "your", "me": "you", "your": "my"},
        "synons": {
            "be": ["am", "is", "are", "was", "were"],
            "joy": ["happy", "glad", "better", "fine"],
            "sadness": ["sad", "depressed", "sick", "gloomy"],
        },
        "keywords": [
            # [Regex, [Responses], Rank]
            [r"(.*) die (.*)", ["Please don't talk like that. Tell me more about your feelings."], 10],
            [r"i need (.*)", ["Why do you need {0}?", "Would it help you to get {0}?"], 5],
            [r"i am (.*)", ["Is it because you are {0} that you came to me?", "How long have you been {0}?"], 5],
            [r"(.*) problem (.*)", ["Tell me more about this problem.", "How does it make you feel?"], 8],
            [r"(.*)", ["Please tell me more.", "I see.", "Can you elaborate?"], 0],
        ],
    },
    "mya": {
        "initials": ["မင်္ဂလာပါ။ သင့်ပြဿနာကို ပြောပြပါ။", "ဘာကြောင့် စိတ်ပူနေပါသလဲ?"],
        "finals": ["နောက်တစ်ခေါက်ပြန်ဆုံမယ်။", "ဆက်သွယ်ပေးတဲ့အတွက် ကျေးဇူးတင်ပါတယ်။"],
        "quits": ["bye", "quit", "exit", "ထွက်", "ထွက်မယ်", "ဘိုင်"],
        "pres": {"ကျွန်မ": "ကျွန်တော်", "ကၽြန္မ": "ကၽြန္ေတာ္"},
        "posts": {"ကျွန်တော်": "သင်", "ငါ": "သင်", "ကျွန်မ": "သင်"},
        "synons": {
            "joy": ["ပျော်", "ဝမ်းသာ"],
            "sadness": ["ဝမ်းနည်း", "စိတ်မကောင်း"],
            "anger": ["စိတ်ဆိုး", "ဒေါသ"],
        },
        "keywords": [
            # [Regex, [Responses], Rank]
            [r"(.*)သေ(.*)", ["အဲဒီလို မပြောပါနဲ့။ သင့်ခံစားချက်တွေကို ပြောပြပါ။"], 10],
            [r"(.*)လိုအပ်(.*)", ["ဘာကြောင့် {0} လိုအပ်တာလဲ?", "{0} ရရင် ကူညီနိုင်မလား?"], 5],
            [r"(.*)ပြဿနာ(.*)", ["ဒီပြဿနာအကြောင်း ပိုပြောပြပါ။", "သင့်ကို ဘယ်လိုခံစားစေသလဲ?"], 8],
            [r"(.*)", ["ပိုပြောပြပါ။", "နားလည်တယ်။", "အသေးစိတ်ပြောပြလို့ ရမလား?"], 0],
        ],
    },
}


def run_oppaword_cli(
    input_path,
    output_path,
    dict_path,
    arpa_path=None,
    space_remove_mode="my_not_num",
    use_bimm_fallback=True,
    bimm_boost=150,
    script_path=DEFAULT_OPPAWORD_SCRIPT,
):
    cmd = ["python3", script_path, "--input", input_path, "--dict", dict_path, "--space-remove-mode", space_remove_mode]
    if arpa_path:
        cmd += ["--arpa", arpa_path]
    if use_bimm_fallback:
        cmd += ["--use-bimm-fallback"]
    if bimm_boost is not None:
        cmd += ["--bimm-boost", str(bimm_boost)]
    cmd += ["--output", output_path]
    subprocess.run(cmd, check=True)


def run_myword_build_dict(dataset_path, script_path="myword.py"):
    cmd = ["python3", script_path, "build_dict", dataset_path]
    subprocess.run(cmd, check=True)


def run_myword_word(input_path, output_path, script_path="myword.py"):
    cmd = ["python3", script_path, "word", input_path, output_path]
    subprocess.run(cmd, check=True)


# --- 2. NEURAL ENGINE COMPONENTS ---
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word2id, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word2id = word2id
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(str(self.texts[idx]))
        seq = [self.word2id.get(w, 1) for w in tokens][: self.max_len]
        padding = [0] * (self.max_len - len(seq))
        return torch.tensor(seq + padding), torch.tensor(self.labels[idx], dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1), weights


class EmotionalBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return self.fc(context)


# --- 3. THE HYBRID CONTROLLER ---
class HybridEliza:
    def __init__(
        self,
        lang="mya",
        model_path=None,
        tokenizer_name="mmdt",
        myword_dict=None,
        oppaword_script=DEFAULT_OPPAWORD_SCRIPT,
        oppaword_dict=None,
        oppaword_arpa=None,
        oppaword_space_remove_mode="my_not_num",
        oppaword_use_bimm_fallback=True,
        oppaword_bimm_boost=150,
    ):
        self.lang = lang
        self.tokenizer_name = tokenizer_name
        self.myword_dict = myword_dict
        # Sort keywords by Rank (index 2) descending immediately
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)

        self.model_path = model_path or f"eliza_eq_{lang}.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        # Updated Kaggle-compliant mapping
        self.id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        self.model = None
        self.oppaword_script = oppaword_script
        self.oppaword_dict = oppaword_dict
        self.oppaword_arpa = oppaword_arpa
        self.oppaword_space_remove_mode = oppaword_space_remove_mode
        self.oppaword_use_bimm_fallback = oppaword_use_bimm_fallback
        self.oppaword_bimm_boost = oppaword_bimm_boost
        self._tokenize_fn = self._build_tokenizer()

    def _build_tokenizer(self):
        if self.lang == "en":
            return lambda text: text.lower().split()

        if self.tokenizer_name == "mmdt":
            return self._init_mmdt()
        if self.tokenizer_name == "oppaword":
            return self._init_oppaword()
        if self.tokenizer_name == "myword":
            return self._init_myword()

        raise ValueError("Unknown tokenizer. Use mmdt, oppaword, or myword.")

    def _flatten_tokens(self, tokens):
        flat = []
        for t in tokens:
            if isinstance(t, list):
                flat.extend([str(x) for x in t])
            else:
                flat.append(str(t))
        return [t for t in flat if t]

    def _init_mmdt(self):
        try:
            from mmdt_tokenizer import MyanmarTokenizer
        except Exception as exc:
            raise RuntimeError("mmdt-tokenizer not installed. Run: pip install mmdt-tokenizer") from exc
        tokenizer = MyanmarTokenizer()
        return lambda text: self._flatten_tokens(tokenizer.word_tokenize(text))

    def _oppaword_tokenize(self, text):
        if not self.oppaword_dict:
            raise RuntimeError("oppaWord requires --oppaword_dict path. Example: --oppaword_dict data/myg2p_mypos.dict")
        fin_path = None
        fout_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as fin:
                fin.write(text)
                fin_path = fin.name
            with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8", delete=False) as fout:
                fout_path = fout.name
            cmd = [
                "python3",
                self.oppaword_script,
                "--input",
                fin_path,
                "--dict",
                self.oppaword_dict,
                "--space-remove-mode",
                self.oppaword_space_remove_mode,
            ]
            if self.oppaword_arpa:
                cmd += ["--arpa", self.oppaword_arpa]
            if self.oppaword_use_bimm_fallback:
                cmd += ["--use-bimm-fallback"]
            if self.oppaword_bimm_boost is not None:
                cmd += ["--bimm-boost", str(self.oppaword_bimm_boost)]
            cmd += ["--output", fout_path]
            subprocess.run(cmd, check=True)
            with open(fout_path, "r", encoding="utf-8") as f:
                out = f.read().strip()
            return out.split() if out else []
        finally:
            for path in (fin_path, fout_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _init_oppaword(self):
        return lambda text: self._flatten_tokens(self._oppaword_tokenize(text))

    def _init_myword(self):
        try:
            from myword import WordTokenizer
        except Exception:
            WordTokenizer = None
        if WordTokenizer:
            # Allow local dict override to avoid HF downloads (CLI flag or env)
            dict_dir = self.myword_dict or os.environ.get("MYWORD_DICT_DIR")
            if dict_dir:
                try:
                    import myword.tokenizer as _mw_tok
                    _mw_tok.HF_myWord_DICT_REPO = dict_dir
                except Exception:
                    pass
            tok = WordTokenizer()
            return lambda text: self._flatten_tokens(tok.tokenize(text))

        candidates = [
            ("myword", "word_tokenize"),
            ("myword", "tokenize"),
            ("myWord", "word_tokenize"),
            ("myWord", "tokenize"),
        ]
        for module_name, fn_name in candidates:
            try:
                mod = __import__(module_name, fromlist=[fn_name])
                fn = getattr(mod, fn_name)
                return lambda text: self._flatten_tokens(fn(text))
            except Exception:
                continue
        raise RuntimeError("myWord tokenizer not found. Install the correct package or adjust the import list.")

    def _tokenize(self, text):
        return self._tokenize_fn(text)

    def build_vocab(self, texts):
        words = Counter([w for t in texts for w in self._tokenize(str(t))])
        for i, (w, _) in enumerate(words.most_common(5000), 2):
            self.word2id[w] = i

    def _read_csv(self, data_path):
        try:
            return pd.read_csv(data_path)
        except pd.errors.ParserError:
            return pd.read_csv(data_path, engine="python", on_bad_lines="skip")

    def train(self, data_path, epochs, lr, batch_size, val_split=0.1, test_split=0.1, eval_report=False, eval_matrix=False):
        df = self._read_csv(data_path)
        label_col = "label" if "label" in df.columns else "emotions"

        # Clean labels: coerce to numeric, drop NaN/inf, then cast to int
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["text", label_col]).reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)

        # ADDED: clean train/val/test split from the same input dataset
        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.0")

        self.build_vocab(df["text"])
        full_dataset = EmotionDataset(df["text"].tolist(), df[label_col].tolist(), self.word2id, self._tokenize)

        val_size = int(len(full_dataset) * val_split)
        test_size = int(len(full_dataset) * test_split)
        train_size = len(full_dataset) - val_size - test_size

        # ADDED: split into train / val / test
        train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"[*] Training on {self.device} (6 Classes)...")
        print(f"[*] Split sizes -> train: {train_size}, val: {val_size}, test: {test_size}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Original behavior kept: epoch-level validation accuracy
            val_acc = self.evaluate(val_loader)
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}"
            )

        # ADDED: final test evaluation after training finishes
        final_test_acc, report, matrix = self.evaluate_detailed(test_loader)

        print(f"\n[Final Test Accuracy]: {final_test_acc:.2%}")

        if eval_report:
            print("\n[Test Classification Report]")
            print(report)

        if eval_matrix:
            print("\n[Test Confusion Matrix]")
            print(matrix)

        torch.save({"state": self.model.state_dict(), "vocab": self.word2id}, self.model_path)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    # ADDED: detailed evaluation returning accuracy, classification report, and confusion matrix
    def evaluate_detailed(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / len(all_labels)

        report = classification_report(
            all_labels,
            all_preds,
            digits=2
        )

        matrix = confusion_matrix(all_labels, all_preds)

        return acc, report, matrix

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.word2id = checkpoint["vocab"]
            self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6).to(self.device)
            self.model.load_state_dict(checkpoint["state"])
            self.model.eval()

    def get_eq(self, text):
        if not self.model:
            return "Neutral", 0.0
        tokens = self._tokenize(text)[:50]
        token_ids = [self.word2id.get(w, 1) for w in tokens]
        token_ids += [0] * (50 - len(token_ids))
        with torch.no_grad():
            output = self.model(torch.tensor([token_ids]).to(self.device))
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs).item()
            return self.id2label[idx], probs[0][idx].item()

    # ADDED: one-shot inference helper for CLI usage
    def infer_text(self, text):
        emotion, score = self.get_eq(text)
        return {
            "text": text,
            "predicted_label": emotion,
            "confidence": score,
        }

    def rule_respond(self, text):
        text = text.lower()
        for k, v in self.script["pres"].items():
            text = text.replace(k, v)
        # Because keywords were sorted by rank in __init__, we take the first match
        for pattern, resps, rank in self.script["keywords"]:
            match = re.search(pattern, text)
            if match:
                resp = random.choice(resps)
                frags = [self.reflect(g) for g in match.groups() if g]
                return resp.format(*frags) if frags else resp
        return "Please continue."

    def reflect(self, fragment):
        return " ".join([self.script["posts"].get(w, w) for w in fragment.split()])


# --- 4. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="mya", choices=["en", "mya"])
    # ADDED: include "infer" mode for one-shot prediction
    parser.add_argument("--mode", default="train", choices=["chat", "train", "infer"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer", default="mmdt", choices=["mmdt", "oppaword", "myword"])
    parser.add_argument("--myword-dict", default=None, help="Path to local myword dict_ver1 (avoids HF download)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    # ADDED: explicit held-out test split for final model evaluation
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--oppaword_script", default=DEFAULT_OPPAWORD_SCRIPT)
    parser.add_argument("--oppaword_dict", default=DEFAULT_OPPAWORD_DICT)
    parser.add_argument("--oppaword_arpa", default=None)
    parser.add_argument("--oppaword_space_remove_mode", default="my_not_num")
    parser.add_argument("--oppaword_use_bimm_fallback", action="store_true")
    parser.add_argument("--oppaword_bimm_boost", type=int, default=150)
    # ADDED: optional final test report / confusion matrix output
    parser.add_argument("--eval_report", action="store_true", help="Print classification report on test split after training")
    parser.add_argument("--eval_matrix", action="store_true", help="Print confusion matrix on test split after training")
    # ADDED: input text for one-shot inference mode
    parser.add_argument("--infer_text", default=None, help="Input text for one-shot inference")
    args = parser.parse_args()

    if args.data is None:
        args.data = "emotions.csv" if args.lang == "en" else "emotions_mya.csv"

    eliza = HybridEliza(
        lang=args.lang,
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        myword_dict=args.myword_dict,
        oppaword_script=args.oppaword_script,
        oppaword_dict=args.oppaword_dict,
        oppaword_arpa=args.oppaword_arpa,
        oppaword_space_remove_mode=args.oppaword_space_remove_mode,
        oppaword_use_bimm_fallback=args.oppaword_use_bimm_fallback,
        oppaword_bimm_boost=args.oppaword_bimm_boost,
    )

    if args.mode == "train":
        eliza.train(
            args.data,
            args.epochs,
            0.001,
            args.batch_size,
            args.val_split,
            args.test_split,
            eval_report=args.eval_report,
            eval_matrix=args.eval_matrix,
        )

    # ADDED: model-only one-shot inference branch
    elif args.mode == "infer":
        eliza.load_model()

        # ADDED: require inference text for infer mode
        if args.infer_text is None:
            raise ValueError("Please provide --infer_text for inference mode.")

        result = eliza.infer_text(args.infer_text)
        print("\n[Inference Result]")
        print(f"Text: {result['text']}")
        print(f"Predicted Emotion: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")

    else:
        eliza.load_model()
        print(f"ELIZA: {random.choice(SCRIPTS[args.lang]['initials'])}")
        while True:
            try:
                user_in = input("You: ")
                if user_in.lower() in SCRIPTS[args.lang]["quits"]:
                    break
                resp = eliza.rule_respond(user_in)
                emotion, score = eliza.get_eq(user_in)
                print(f"ELIZA: {resp}")
                print(f"[EQ Analysis]: Predicted Emotion: {emotion} ({score:.2%})")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()