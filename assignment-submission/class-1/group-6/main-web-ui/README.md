# 🇲🇲 Myanmar Elyza: Emotion-Aware Chatbot

A specialized Myanmar language chatbot that integrates **Deep Learning (BiLSTM + Attention)** for emotion detection with a **Rule-Based engine** for empathetic conversational flow.

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| `app_cleaned.py` | Main application (Streamlit UI + Rule-Based engine) |
| `Deploy.ipynb` | Google Colab deployment notebook |
| `eliza_my.pth` | Pre-trained PyTorch model weights |

---

## Getting Started

### 1. Requirements

Install the following dependencies:

- `streamlit`
- `torch`
- `pyngrok`

---

## Run the Project

### Run Locally

```bash
# Install dependencies
pip install streamlit torch pyngrok

# Run the app
streamlit run app_cleaned.py
```
---
## Technical Details

### Emotion Detection Architecture
The core of the "Myanmar Elyza" intelligence is a **Deep Learning** model implemented in **PyTorch**, specifically designed to handle the nuances of the Myanmar language.

*   **Input Layer:** Processes normalized Myanmar Unicode text.
*   **BiLSTM Layer (Bidirectional Long Short-Term Memory):** Processes the text sequence in both forward and backward directions to capture full contextual meaning.
*   **Attention Mechanism:** A custom `Attention` class that assigns "weights" to specific words. This allows the model to focus on emotionally charged Burmese keywords (like *စိတ်မကောင်း* or *ဝမ်းသာ*) regardless of where they appear in the sentence.
*   **Output Layer:** A Linear layer that maps the hidden states to a probability distribution across **6 emotion labels**: *Sadness, Joy, Love, Anger, Fear, and Surprise*.

### 🛠️ Response Priority System
The chatbot does not just reply randomly; it follows a **Hierarchical Rank System** defined in the logic to ensure user safety and conversational flow:

| Priority Level | Rank | Purpose | Example Trigger |
| :--- | :--- | :--- | :--- |
| **Critical Safety** | **120** | Detects mentions of self-harm or emergencies to provide immediate support. | *သေချင်တယ်* (Want to die) |
| **Input Validation** | **100** | Detects if the message is purely English and politely requests Myanmar text. | "Hello, how are you?" |
| **Interaction** | **98 - 97** | Handles messages that are only Emojis or symbols. | "😊😊😊" |
| **Medical/Grief** | **95** | Provides specialized comfort for physical illness or deep loss. | *နေမကောင်းဘူး* (Not feeling well) |
| **Emotion-Based** | **70 - 90** | Tailors responses to the user's detected mood (Joy, Love, Stress). | *ပင်ပန်းနေပြီ* (I'm tired/stressed) |
| **Catch-all** | **0** | Default responses to keep the dialogue active. | *ဆက်ပြောပါ* (Please continue) |
