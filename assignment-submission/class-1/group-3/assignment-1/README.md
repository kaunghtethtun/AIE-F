# Hybrid ELIZA for Burmese Emotion Analysis

This project integrates a rule-based ELIZA chatbot with a Neural Engine (BiLSTM with Attention) to detect and respond to emotions in the Burmese language. It classifies inputs into 6 emotions (Sadness, Joy, Love, Anger, Fear, Surprise) and provides context-aware chatbot responses.

## 📂 Project Structure

- `hybrid-eliza.py`: The main script containing the Hybrid ELIZA chatbot and the PyTorch BiLSTM model for training and chatting.
- `test.py`: The evaluation script that runs inference on a held-out test set and generates performance metrics (F1, Precision, Recall, Accuracy) and a Confusion Matrix.
- `split_data.sh`: A shell script to split the initial dataset into training (2/3) and testing (1/3) sets.
- `run.sh`: An automated pipeline script that trains the model, evaluates it, and starts an interactive chat session.
- `data/`: Directory containing the target CSV dataset files.
- `logs/`: Directory where training and chat logs are saved.
- `models/`: Directory where trained PyTorch models are saved.
- `assets/`: Directory where generated charts and evaluation metrics (e.g. confusion matrix) are saved.

---

## 🚀 Quick Start

### 1. Split the Data
Before training, prepare your `train.csv` and `test.csv` using the automated split script:

```bash
./split_data.sh [input_csv] [train_csv] [test_csv]
```

### 2. Run the Pipeline (Train, Test, and Chat)
To execute the complete end-to-end pipeline (Training -> Evaluation -> Interactive Chat):
```bash
./run.sh
```

---

## 📖 Usage Alternatives

If you wish to run the components individually instead of using `run.sh`, you can use the following commands:

**To Train the Model Manually:**
```bash
python hybrid-eliza.py --lang my --mode train --data data/train.csv --epochs 4 --val_split 0.2 --model_path ./models/eliza_mm2.pth
```

**To Evaluate the Trained Model:**
```bash
python test.py --model ./models/eliza_mm2.pth --test_data data/test.csv
```

**To Start the Chatbot (Requires Trained Model):**
```bash
python hybrid-eliza.py --lang my --mode chat --model_path ./models/eliza_mm2.pth
```

---

## ⚠️ Disclaimer & Prerequisites
- **Data Folder:** Ensure you have your source dataset `cleaned_burmese_emotion_data.csv` placed in the `data/` folder before running data splitting. Ensure `train.csv` and `test.csv` are present in `data/` before running the training script (`run.sh`).
- **Frameworks:** Ensure you have Python 3 installed along with `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.