# Hybrid-ELIZA Multi (EN + Myanmar) — Final Version

This document provides instructions for running the final implementation:

```
hybrid-eliza-multi-final.py
```

## Features

* Train / validation / test data split
* Final test evaluation (accuracy, classification report, confusion matrix)
* Multi-tokenizer support (mmdt, oppaword, myword)
* Inference mode for single input prediction
* Chat mode combining ELIZA rules with emotion classification

---

# 1. Overview

This system is a hybrid architecture combining:

* Rule-based ELIZA for conversational responses
* BiLSTM with Attention for emotion classification

## Supported Languages

* English (`en`)
* Myanmar (`mya`)

---

# 2. Installation

## Python Environment

Ensure Python 3 is installed:

```bash
python3 --version
```

## Dependencies

```bash
pip install torch pandas numpy scikit-learn
```

---

# 3. Tokenizers (Myanmar Only)

Three tokenizer options are supported:

## 3.1 mmdt (Recommended for simplicity)

```bash
pip install mmdt-tokenizer
```

## 3.2 oppaWord

```bash
git clone https://github.com/ye-kyaw-thu/oppaWord.git
```

## 3.3 myWord

```bash
git clone https://github.com/ye-kyaw-thu/myWord.git
```

---

# 4. Dataset Format

Input data must be in CSV format:

```csv
text,label
ငါ ဝမ်းနည်း နေတယ်,0
ငါ ပျော် တယ်,1
```

## Label Mapping

```
0 = Sadness  
1 = Joy  
2 = Love  
3 = Anger  
4 = Fear  
5 = Surprise  
```

---

# 5. Training

## 5.1 Basic Training (Myanmar with mmdt)

```bash
python3 hybrid-eliza-multi-final.py \
  --mode train \
  --lang mya \
  --data final.csv \
  --tokenizer mmdt
```

## 5.2 Training with oppaWord

```bash
python3 hybrid-eliza-multi-final.py \
  --mode train \
  --lang mya \
  --data final.csv \
  --tokenizer oppaword \
  --oppaword_script ./oppaWord/oppa_word.py \
  --oppaword_dict ./oppaWord/data/myg2p_mypos.dict
```

## 5.3 Training with Full Evaluation

```bash
python3 hybrid-eliza-multi-final.py \
  --mode train \
  --lang mya \
  --data final.csv \
  --tokenizer oppaword \
  --oppaword_script ./oppaWord/oppa_word.py \
  --oppaword_dict ./oppaWord/data/myg2p_mypos.dict \
  --val_split 0.1 \
  --test_split 0.1 \
  --eval_report \
  --eval_matrix
```

---

# 6. Output Interpretation

## Training Output

```
Epoch X | Loss | Val Acc
```

## Final Test Performance

```
[Final Test Accuracy]: 56.48%
```

## Classification Report

Displays precision, recall, and F1-score for each class.

## Confusion Matrix

```
[[...], [...], ...]
```

This represents the final model performance on the test set.

---

# 7. Inference (Single Input)

## 7.1 Myanmar Example

```bash
python3 hybrid-eliza-multi-final.py \
  --mode infer \
  --lang mya \
  --model_path eliza_eq_mya.pth \
  --tokenizer oppaword \
  --oppaword_script ./oppaWord/oppa_word.py \
  --oppaword_dict ./oppaWord/data/myg2p_mypos.dict \
  --infer_text "ငါ ဝမ်းနည်း နေတယ်"
```

### Output

```
Predicted Emotion: Sadness  
Confidence: 78%
```

---

## 7.2 English Example

```bash
python3 hybrid-eliza-multi-final.py \
  --mode infer \
  --lang en \
  --model_path eliza_eq_en.pth \
  --infer_text "i feel really happy today"
```

---

# 8. Chat Mode

```bash
python3 hybrid-eliza-multi-final.py \
  --mode chat \
  --lang mya \
  --tokenizer mmdt
```

### Example

```
You: ငါ ဝမ်းနည်း နေတယ်  
ELIZA: ပိုပြောပြပါ။  
[EQ Analysis]: Predicted Emotion: Sadness (82%)
```

---

# 9. Model Architecture

* Embedding layer
* Bidirectional LSTM
* Attention layer
* Fully connected output layer (6 classes)

---

# 10. Key Characteristics

* Multi-language support (English and Myanmar)
* Flexible tokenizer integration
* Hybrid rule-based and neural approach
* Proper machine learning workflow (train → validation → test)
* Command-line inference support

---

# 11. Notes

* Stopword removal is not applied, as it may affect emotion detection
* ZWNJ/ZWSP cleaning is optional preprocessing
* Dataset quality (spelling consistency, length variation) impacts performance

Current performance (~56%) is reasonable given:

* Limited dataset size
* Six-class classification problem
* Low-resource language setting

---

# 12. Project Structure

```
project/
│
├── hybrid-eliza-multi-final.py
├── final.csv
├── eliza_eq_mya.pth
├── oppaWord/
└── myWord/
```

---

# 13. Summary

This project demonstrates:

* A hybrid conversational AI system
* Burmese NLP preprocessing and modeling
* Emotion classification using BiLSTM with attention
* End-to-end machine learning pipeline including deployment-ready inference

---

# 14. Streamlit Chat Interface (No Retraining Required)

A Streamlit interface is provided to interact with the trained model without retraining.

## Execution

```bash
streamlit run app.py
```

Access the interface at:

```
http://localhost:8501
```

---

## 14.1 System Flow

```
Load trained model (.pth)
→ Run inference
→ Generate ELIZA response
→ Display emotion prediction
```

---

## 14.2 Core Mechanism

### Dynamic Model Loading

```python
load_eliza(model_path)
```

* Automatically determines language
* Loads tokenizer configuration
* Loads trained model

---

### Tokenizer Handling

For Myanmar:

```python
tokenizer_name = "oppaword"
```

Required paths:

```
oppaWord/oppa_word.py  
oppaWord/data/myg2p_mypos.dict
```

---

### Dynamic Import

```python
importlib.util.spec_from_file_location(...)
```

Allows direct loading of Python modules without installation.

---

### Chat State Management

```python
st.session_state.messages
```

Message format:

```python
{"role": "user" | "assistant", "content": "..."}
```

---

## 14.3 Chat Flow

```
User input
→ ELIZA response
→ Emotion prediction
→ Combined output
```

---

## 14.4 Example

Input:

```
ငါ ဝမ်းနည်း နေတယ်
```

Output:

```
ပိုပြောပြပါ။

[EQ Analysis] Predicted Emotion: Sadness (78%)
```

---

## 14.5 Reset Function

```python
st.button("Reset Chat")
```

Resets conversation state and initializes a new session.

---

## 14.6 Notes

* Requires a trained `.pth` model
* oppaWord paths must be valid
* Tokenizer must match training configuration

---
