## Hybrid-ELIZA (EN + Myanmar)
## Demo code of applying rules + BiLSTM for AI Engineering Class (Fundamental)
## Written by Ye, Language Understanding Lab., Myanmar
## Last updated: 19 Mar 2026
## Reference code: https://www.kaggle.com/code/wjburns/eliza

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import argparse
import random

try:
    # Optional: oppaWord-inspired tokenizer (see mmdt-tokenizer)
    from mmdt_tokenizer import MyanmarTokenizer
except Exception:
    MyanmarTokenizer = None

# --- 1. GLOBAL SCRIPT DATA ---
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
        "initials": ["မင်္ဂလာပါ။ ဘာအကြောင်း ပြောချင်လဲ?", 
                     "ဟုတ်ကဲ့... ဘာဖြစ်နေလဲ ပြောပြပါ။",
                     "စိတ်ထဲမှာ ဘာတွေရှိလဲ ပြောပြလို့ရပါတယ်။"],
        "finals": ["ဒီနေ့ပြောခဲ့တာတွေ ကျေးဇူးတင်ပါတယ်။ နောက်မှပြန်ဆုံမယ်နော်။",
                    "ကောင်းပါတယ်။ ကိုယ့်ကိုယ်ကို ဂရုစိုက်ပါနော်။",],
        "quits": ["bye", "quit", "exit", "ထွက်", "ထွက်မယ်", "ဘိုင်"],
        "pres": {"ကျွန်မ": "ကျွန်တော်", "ကၽြန္မ": "ကၽြန္ေတာ္"},
        "posts": {"ကျွန်တော်": "သင်", "ငါ": "သင်", "ကျွန်မ": "သင်"},
        "synons": {
            "joy": ["ပျော်", "ဝမ်းသာ", "နေကောင်း"],
            "sadness": ["ဝမ်းနည်း", "စိတ်မကောင်း", "စိတ်ညစ်"],
            "anger": ["စိတ်ဆိုး", "ဒေါသ", "မကျေနပ်"],
        },
        "keywords": [
        # High priority
        [r"(.*)သေ(.*)", [
            "အဲဒီလို မပြောပါနဲ့နော်။ ဘာကြောင့် အဲလိုခံစားနေရတာလဲ?",
            "စိတ်ထဲမှာ ဘာတွေဖြစ်နေလဲ ပြောပြပါ။"
        ], 10],

        # Need
        [r"(.*)လိုအပ်(.*)", [
            "{0} လိုအပ်တယ်ဆိုတာ ဘာကြောင့်လဲ?",
            "{0} ရရင် သင့်အတွက် ဘယ်လိုကူညီမလဲ?"
        ], 6],

        # Feeling
        [r"(.*)(ဝမ်းနည်း|စိတ်မကောင်း)(.*)", [
            "ဘာကြောင့် အဲလိုခံစားနေရတာလဲ?",
            "အဲဒီအကြောင်းကို နည်းနည်းပိုပြောပြပါလား။"
        ], 8],

        [r"(.*)(ပျော်|ဝမ်းသာ)(.*)", [
            "ကောင်းတာပဲ။ ဘာကြောင့် ပျော်နေတာလဲ?",
            "အဲဒီကောင်းတဲ့အရာကို မျှဝေပေးပါ။"
        ], 7],

        # Problem
        [r"(.*)ပြဿနာ(.*)", [
            "အဲဒီပြဿနာကို နည်းနည်းပိုရှင်းပြပါလား။",
            "အဲဒါက သင့်ကို ဘယ်လိုအကျိုးသက်ရောက်စေလဲ?"
        ], 8],

        # Generic fallback
        [r"(.*)", [
            "နည်းနည်းပိုပြောပြပါလား။",
            "ဟုတ်ကဲ့... နားထောင်နေပါတယ်။",
            "အဲဒါကို နည်းနည်းပိုရှင်းပြလို့ရမလား?"
        ], 0],
    ],
    },
}

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
    def __init__(self, lang="mya", model_path=None, tokenizer_name="mmdt"):
        self.lang = lang
        self.tokenizer_name = tokenizer_name
        # Sort keywords by Rank (index 2) descending immediately
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)

        self.model_path = model_path or f"eliza_eq_{lang}.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        # Updated Kaggle-compliant mapping
        self.id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        self.model = None
        self.mya_tokenizer = None
        if self.lang == "mya" and self.tokenizer_name == "mmdt":
            if not MyanmarTokenizer:
                raise RuntimeError("mmdt-tokenizer not installed. Run: pip install mmdt-tokenizer")
            self.mya_tokenizer = MyanmarTokenizer()

    def clean_text(self, text):
        if self.lang == "mya":
            # Keep Myanmar letters and spaces. Normalize whitespace.
            cleaned = re.sub(r"[^\u1000-\u109F\s]", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _tokenize(self, text):
        cleaned = self.clean_text(text)
        if self.lang == "en":
            return cleaned.split() if cleaned else []
        if self.tokenizer_name == "mmdt":
            tokens = self.mya_tokenizer.word_tokenize(cleaned)
            # Some tokenizers return nested lists; flatten and stringify safely.
            flat = []
            for t in tokens:
                if isinstance(t, list):
                    flat.extend([str(x) for x in t])
                else:
                    flat.append(str(t))
            return [t for t in flat if t]
        if self.tokenizer_name == "space":
            return cleaned.split() if cleaned else []
        if self.tokenizer_name == "char":
            return [ch for ch in cleaned.replace(" ", "") if ch]
        return []

    def build_vocab(self, texts):
        words = Counter([w for t in texts for w in self._tokenize(str(t))])
        for i, (w, _) in enumerate(words.most_common(5000), 2):
            self.word2id[w] = i

    def train(self, data_path, epochs, lr, batch_size, val_split=0.1):
        try:
            df = pd.read_csv(data_path)
        except pd.errors.ParserError:
            # Fallback for CSVs with stray commas in text fields
            df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
        label_col = "label" if "label" in df.columns else "emotions"
        # Clean labels: coerce to numeric, drop NaN/inf, then cast to int
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["text", label_col]).reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)

        self.build_vocab(df["text"])
        full_dataset = EmotionDataset(df["text"].tolist(), df[label_col].tolist(), self.word2id, self._tokenize)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"[*] Training on {self.device} (6 Classes)...")
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

            # Epoch Evaluation
            val_acc = self.evaluate(val_loader)
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}"
            )

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
        tokens = self._tokenize(fragment)
        return " ".join([self.script["posts"].get(w, w) for w in tokens])


# --- 4. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="mya", choices=["en", "mya"])
    parser.add_argument("--mode", default="train", choices=["chat", "train"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer", default="mmdt", choices=["mmdt", "space", "char"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    if args.data is None:
        args.data = "emotions.csv" if args.lang == "en" else "emotions_mya.csv"

    eliza = HybridEliza(lang=args.lang, model_path=args.model_path, tokenizer_name=args.tokenizer)

    if args.mode == "train":
        eliza.train(args.data, args.epochs, 0.001, args.batch_size, args.val_split)
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
