import streamlit as st
import torch
import torch.nn as nn
import re
import random
import base64
import html as htmllib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="မြန်မာဘာသာ Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ─────────────────────────────────────────────
# DATA & SCRIPTS
# ─────────────────────────────────────────────
SCRIPTS = {
    "my": {
        "initials": [
            "မင်္ဂလာပါခင်ဗျာ။ ဘာများ အခက်အခဲရှိလို့လဲဟင်? ပြောပြလို့ရပါတယ်။",
            "မင်္ဂလာပါ။ ဘာများ ကူညီပေးရမလဲခင်ဗျာ?"
        ],
        "finals": [
            "နောက်မှ ပြန်ဆုံကြတာပေါ့။ ဂရုစိုက်ပါနော်။",
            "စကားပြောရတာ ဝမ်းသာပါတယ်။ စိတ်သက်သာရာရပါစေနော်။"
        ],
        # FIX 1: Added colloquial quit words + "cu"
        "quits": [
    "ဘိုင်", "သွားပြီ",
    "တာ့တာ", "ဘိုင့်",
    "သွားမယ်", "သွားတော့မယ်", "နှုတ်ဆက်",
    "လိုက်တော့မယ်", "ပြီးပြီ", "ရပ်လိုက်",
    "bye", "quit", "exit", "goodbye", "see you", "cya", "cu"
],

        "pres": {
            "မဟုတ်ဘူး": "မဟုတ်",
            "မလုပ်ဘူး": "မလုပ်",
            "မသိဘူး": "မသိ",
        },

        "posts": {
            "ကျွန်တော်":  "သင်",
            "ကျွန်မ":     "သင်",
            "ငါ":         "သင်",
            "ငါ့":        "သင့်",
            "ကျွန်တော့်": "သင့်",
            "ကျွန်မ့်":   "သင့်",
            "သင်":        "ကျွန်တော်",
            "သင့်":       "ကျွန်တော့်",
        },

       "keywords": [
    # --- LEVEL 1: CRITICAL SAFETY (Rank 120) ---
    # အသက်အန္တရာယ်နှင့် စိတ်ပိုင်းဆိုင်ရာ အရေးပေါ်အခြေအနေ (Hotline မပါသော်လည်း အလေးအနက်ထားဖြေကြားခြင်း)
    [r"(.*)(သေချင်|ကိုယ့်ကိုယ်ကိုယ်|အဆုံးစီရင်|မရှိတော့ချင်|သေတာပဲကောင်းတယ်)(.*)",
     [
         "ဒီလိုတွေ ခံစားနေရတာ တကယ်ကို ပင်ပန်းနေမှာပဲနော်။ တစ်ယောက်တည်း စိတ်မညစ်ပါနဲ့။ ရင်ဖွင့်ကြည့်ပါဦး။ ကျွန်တော်လည်း ဒီမှာ ရှိနေပေးပါမယ်။",
         "အခုလို ခံစားနေရတာကို ကြားရတာ စိုးရိမ်မိပါတယ်ဗျာ။ စိတ်ကို ဖြေလျှော့ပြီး အကူအညီတောင်းကြည့်ဖို့ တိုက်တွန်းချင်ပါတယ်။ အချိန်မရွေး ရင်ဖွင့်လို့ ရပါတယ်နော်။"
     ], 120],

    [r"(.*)(ရိုက်|နှိပ်စက်|ကြောက်နေရ|အနိုင်ကျင့်|အကြမ်းဖက်)(.*)",
     [
         "အခု ဘေးကင်းရဲ့လားဟင်? ဒီလို အခြေအနေမျိုးက တကယ်ကို ကြောက်ဖို့ကောင်းပါတယ်။ အကူအညီပေးနိုင်မယ့်သူ တစ်ယောက်ယောက်ကို အသိပေးထားဖို့ အရေးကြီးပါတယ်နော်။",
         "ဘေးကင်းလုံခြုံမှုက အရေးကြီးဆုံးပါ။ အနီးအနားမှာ ယုံကြည်ရတဲ့သူ ရှိလားဟင်?"
     ], 110],

    # --- LEVEL 2: INPUT VALIDATION - ENGLISH ONLY (Rank 100) ---
    # စာကြောင်းတစ်ခုလုံး English ဖြစ်နေမှသာ ဖြေကြားရန် (Mixed text ကို ခွင့်ပြုသည်)
    [r"^[a-zA-Z0-9\s\.\,\!\?\'\"\-]+$",
     [
         "ကျွန်တော် မြန်မာလိုပဲ နားလည်လို့ မြန်မာလိုလေး ရေးပေးလို့ ရမလားဟင်? 😊",
         "တောင်းပန်ပါတယ်ခင်ဗျာ။ ကျွန်တော် မြန်မာဘာသာစကားနဲ့ပဲ ပြောဆိုနိုင်လို့ မြန်မာလိုလေး ပြောပေးပါဦးနော်။"
     ], 100],

    # --- LEVEL 3: EMOJI HANDLER (Rank 98) ---
    [r"^[\u2600-\u27BF\U0001f300-\U0001f64f\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]+$",
     [
         "ဒီ Emoji လေးတွေ မြင်ရတာ ဝမ်းသာပါတယ် 😊 ဒါပေမဲ့ စိတ်ထဲရှိတာလေးတွေကို စာနဲ့ ရေးပြလို့ ရမလားဟင်?",
         "Emoji လေးတွေက ချစ်စရာကောင်းလိုက်တာ။ စိတ်ထဲ ဘယ်လို ခံစားနေရလဲဆိုတာ စာနဲ့ ပြောပြပါဦး။"
     ], 98],

    # --- LEVEL 4: SYMBOLS & PUNCTUATION (Rank 97) ---
    [r"^[^a-zA-Z\u1000-\u109F]+$",
     [
         "သင်္ကေတလေးတွေပဲ တွေ့ရတယ်နော် 😊 စကားလေးနဲ့ ရေးပြလို့ ရမလားဟင်?",
         "နားမလည်တဲ့ သင်္ကေတလေးတွေ ဖြစ်နေလို့ စာလေးနဲ့ ပြန်ပြောပေးပါဦးနော်။"
     ], 97],

    # --- LEVEL 5: PHYSICAL HEALTH & GRIEF (Rank 95) ---
    [r"(.*)(နေမကောင်း|မသက်သာ|ဖျား|နာကျင်|နာတယ်|မော|ကိုက်)(.*)",
     [
         "အို... ကြားရတာ စိတ်မကောင်းပါဘူး။ နေမကောင်းရင် အနားယူဖို့ မမေ့နဲ့နော်။ ဘယ်လိုတွေ ဖြစ်နေတာလဲဟင်?",
         "အမြန်ဆုံး သက်သာသွားအောင် ကိုယ့်ကိုယ်ကိုယ် ဂရုစိုက်ပါဦးနော်။"
     ], 95],

    # --- LEVEL 6: EMOTIONS & MENTAL STATES (Rank 70-90) ---
    [r"(.*)(ပျော်|ဝမ်းသာ|အဆင်ပြေ|အောင်မြင်|ကောင်းတယ်)(.*)",
     [
         "ဝမ်းသာလိုက်တာ 😊 ဘာကြောင့် အခုလို ပျော်နေတာလဲဟင်?",
         "ကောင်းလိုက်တာဗျာ။ ဒီလိုလေး ဆက်ပြီး စိတ်ချမ်းသာနေဖို့ ဘာတွေက ကူညီပေးလဲ?",
         "အဲ့လိုကြားရတာ တကယ်ဝမ်းသာပါတယ်။ ပျော်စရာအကြောင်းလေး နည်းနည်းပိုပြောပြပါဦး။"
     ], 90],

    [r"(.*)(ချစ်|မြတ်နိုး|သတိရ|ရင်ခုန်|စွဲလမ်း)(.*)",
     [
         "ချစ်ခြင်းမေတ္တာဟာ အင်မတန်အရေးကြီးတယ်နော်။ ဘယ်သူ့ကို ချစ်တာလဲဟင်?",
         "ချစ်သူရှိတာ ကံကောင်းပါတယ်နော်။ ဒီချစ်ခြင်းက ကိုယ့်ကို ဘယ်လို ခံစားစေသလဲ?",
         "ချစ်တတ်တဲ့ နှလုံးသားရှိတာ ကောင်းလိုက်တာ။ ဆက်ပြောပြပါဦး။"
     ], 88],

    [r"(.*)(အံ့ဩ|မထင်မှတ်|တကယ်လား|အံ့အားသင့်|မျှော်မထားဘဲ)(.*)",
     [
         "မထင်မှတ်ဘဲ ဘာဖြစ်သွားတာလဲ? ဆက်ပြောပြပါဦး။",
         "အံ့ဩစရာကောင်းတဲ့ အကြောင်းလေး ပိုပြောပြပါဦး။",
         "မထင်မှတ်ဘဲ ကြုံလိုက်ရတာ ဘယ်လို ခံစားရသလဲ?"
     ], 87],

    [r"(.*)(စိတ်ဖိစီး|ပင်ပန်း|မနိုင်တော့ဘူး|ဖိအားများ)(.*)",
     [
         "အများကြီး ပင်ပန်းနေပုံရတယ်နော်။ ခဏလေး နားပြီး အသက်ရှူလိုက်ပါဦး။",
         "စိတ်ဖိစီးနေတဲ့အချိန်မှာ တစ်ယောက်တည်း မခံစားပါနဲ့။ ကျွန်တော် နားထောင်ပေးပါ့မယ်။"
     ], 85],

    [r"(.*)(ဝမ်းနည်း|စိတ်မကောင်း|စိတ်ဓာတ်ကျ|မပျော်|ငိုချင်)(.*)",
     [
          "အို... ကြားရတာ စိတ်မကောင်းလိုက်တာ။ ရင်ဖွင့်ချင်ရင် ကျွန်တော် နားထောင်ပေးမယ်နော်။",
          "စိတ်ညစ်နေတာကို ကြားရတာ စိတ်မကောင်းပါဘူး။ ဘာဖြစ်လို့လဲဆိုတာ ဖြည်းဖြည်းချင်း ပြောပြပါဦး။"
     ], 80],
    
    [r"(.*)(ငါ့ကြောင့်|ကျွန်တော့်ကြောင့်|ကျွန်မကြောင့်|ကိုယ့်အပြစ်|နောင်တ)(.*)",
     [
         "ကိုယ့်ကိုယ်ကိုယ် အပြစ်တင်မနေပါနဲ့ဦးနော်။ အရာအားလုံးက ကိုယ့်ကြောင့်ပဲ မဟုတ်နိုင်ပါဘူး။",
         "နောင်တရနေတာလားဟင်? ဖြစ်ခဲ့တာတွေအတွက် ကိုယ့်ကိုယ်ကိုယ် အရမ်းကြီး ဖိအားမပေးပါနဲ့ဦး။"
     ], 75],

    # --- LEVEL 7: SITUATIONAL & ADVICE (Rank 65) ---
    [r"(.*)(ဘာလုပ်ရမလဲ|ဘယ်လိုလုပ်ရမလဲ|လမ်းညွှန်ပေး)(.*)",
     [
         "ခက်ခဲနေမှာပဲနော်။ ကျွန်တော်ကတော့ ကိုယ်တိုင် အဆင်ပြေဆုံးဖြစ်မယ့် နည်းလမ်းကို အတူတူ စဉ်းစားပေးချင်ပါတယ်ဟင်။",
         "အတူတူ အဖြေရှာကြည့်ကြတာပေါ့။ အခုလောလောဆယ် ကိုယ်တိုင် ဘယ်လိုမျိုး အဆင်ပြေမယ်လို့ ထင်လဲဟင်?",
         "ဘယ်လိုလုပ်ရမလဲဆိုတာ အဖြေရှာရခက်နေတာလား? စိတ်ထဲမှာ ဘာတွေလုပ်ဖို့ စဉ်းစားထားလဲ၊ ကျွန်တော့်ကို ပြောပြကြည့်ပါဦး။"
     ], 65],

    # --- LEVEL 8: SOCIAL & COPING (Rank 50-60) ---
    [r"(.*)(အဖေ|အမေ|မိဘ|မောင်နှမ|ညီအစ်ကို)(.*)(ပြဿနာ|စကားများ|နားမလည်)(.*)",
     [
         "အိမ်ကလူတွေနဲ့ နားလည်မှုလွဲတာက တကယ်ကို စိတ်ညစ်ဖို့ကောင်းတယ်နော်။ ဘာတွေ အဆင်မပြေဖြစ်လို့လဲဟင်?"
     ], 60],

    [r"(.*)(ဘုရား|တရား|ကံတရား|ဆုတောင်း)(.*)",
     [
         "ဟုတ်ပါတယ်၊ ဘာသာရေးက စိတ်ခွန်အား အများကြီး ရစေတာပေါ့။ အခုလို အားကိုးရာ ရှိတာ ကောင်းပါတယ်။"
     ], 55],

    [r"(.*)(ကျေးဇူး|thanks|thank you)(.*)",
     [
         "ရပါတယ်နော်၊ အားမနာပါနဲ့ 🤍",
         "ဝမ်းသာပါတယ်။ ကူညီပေးနိုင်တာကိုပဲ ကျေနပ်ပါတယ်။"
     ], 50],

    # --- LEVEL 9: THE CATCH-ALL (Rank 0) ---
    [r"(.*)",
     [
         "ဟုတ်ကဲ့၊ ဆက်ပြောပါဦး။ နားထောင်နေပါတယ်ခင်ဗျာ။",
         "ကျွန်တော် နားထောင်ပေးနေပါတယ်နော်။",
         "အေးအေးဆေးဆေး ဖြည်းဖြည်းချင်း ပြောပြလို့ ရပါတယ်။"
     ], 0],
]
    }
}

EMOTION_EMOJI = {
    "Sadness": "😢",
    "Joy": "😊",
    "Love": "❤️",
    "Anger": "😠",
    "Fear": "😨",
    "Surprise": "😲",
}

ID2LABEL = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# ─────────────────────────────────────────────
# ROBOT AVATAR
# ─────────────────────────────────────────────
AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
<circle cx="60" cy="60" r="58" fill="#1a3a5c"/>
<circle cx="60" cy="50" r="30" fill="#e8f4fd"/>
<circle cx="45" cy="45" r="5" fill="#1a3a5c"/>
<circle cx="75" cy="45" r="5" fill="#1a3a5c"/>
<path d="M45 65 Q60 75 75 65" stroke="#1a3a5c" stroke-width="3" fill="none"/>
</svg>
"""

AVATAR_B64 = base64.b64encode(AVATAR_SVG.encode()).decode()
AVATAR_URI = f"data:image/svg+xml;base64,{AVATAR_B64}"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.chat-wrap{
    background:#ffffff;
    border-radius:15px;
    padding:20px;
    height:450px;
    overflow-y:auto;
    border:1px solid #ddd;
    margin-bottom:20px;
}
.msg-row{
    display:flex;
    gap:8px;
    margin-bottom:15px;
    align-items:flex-start;
}
.msg-row.user{
    flex-direction:row-reverse;
}
.msg-content{
    display:flex;
    flex-direction:column;
    max-width:70%;
}
.msg-row.user .msg-content{
    align-items:flex-end;
}
.msg-row.bot .msg-content{
    align-items:flex-start;
}
.bubble{
    padding:10px 15px;
    border-radius:18px;
    max-width:100%;
    font-size:14px;
    white-space:pre-wrap;
    word-wrap:break-word;
    overflow-wrap:break-word;
}
.bubble.user{
    background:#1a3a5c;
    color:white;
    border-bottom-right-radius:3px;
}
.bubble.bot{
    background:#eaf2fb;
    color:#1a2940;
    border-bottom-left-radius:3px;
}
.bot-av{
    width:35px;
    height:35px;
    border-radius:50%;
    flex-shrink:0;
}
.user-av{
    width:35px;
    height:35px;
    background:#ddd;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-shrink:0;
}
.emotion-badge{
    font-size:11px;
    padding:4px 8px;
    border-radius:10px;
    background:#eee;
    margin-top:5px;
    display:inline-block;
    line-height:1.5;
}
.small-score{
    font-size:11px;
    color:#555;
    margin-top:5px;
    line-height:1.6;
    text-align:right;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NLP UTILS
# ─────────────────────────────────────────────
def normalize_myanmar(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def myanmar_syllable_tokenize(text):
    return list(text)

def apply_pres(text):
    for old, new in SCRIPTS["my"]["pres"].items():
        text = text.replace(old, new)
    return text

def apply_posts(text):
    for old, new in SCRIPTS["my"]["posts"].items():
        text = text.replace(old, new)
    return text

def preprocess_text(text):
    text = normalize_myanmar(text)
    text = apply_pres(text)
    return text

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights

class EmotionalBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# ─────────────────────────────────────────────
# EMOTION SCORING
# ─────────────────────────────────────────────
def detect_emotion_scores(text):
    text = normalize_myanmar(text)

    scores = {
        "Sadness": 0.0,
        "Joy":     0.0,
        "Love":    0.0,
        "Anger":   0.0,
        "Fear":    0.0,
        "Surprise":0.0,
    }

    sadness_words  = ["ဝမ်းနည်း","စိတ်မကောင်း","စိတ်ဓာတ်ကျ","မပျော်","ငိုချင်","လွမ်း","နေမကောင်း","မသက်သာ","ဖျား","နာ","မော","အထီးကျန်"]
    joy_words      = ["ပျော်","ဝမ်းသာ","ကောင်းတယ်","အဆင်ပြေ","ပီတိ"]
    love_words     = ["ချစ်","သတိရ","ရင်ခုန်","မြတ်နိုး"]
    anger_words    = ["ဒေါသ","စိတ်ဆိုး","မကျေနပ်","စိတ်တို","အသည်းအသန်"]
    fear_words     = ["ကြောက်","စိုးရိမ်","ပူပန်","မသေချာ","မပြောချင်","မဖြေချင်","စိတ်ဖိစီး","ဖိအားများ","မနိုင်တော့ဘူး"]
    surprise_words = ["အံ့ဩ","မထင်ထား","တကယ်လား","ဟင်","အံ့အားသင့်"]

    for w in sadness_words:
        if w in text: scores["Sadness"] += 2.0
    for w in joy_words:
        if w in text: scores["Joy"] += 2.0
    for w in love_words:
        if w in text: scores["Love"] += 2.0
    for w in anger_words:
        if w in text: scores["Anger"] += 2.0
    for w in fear_words:
        if w in text: scores["Fear"] += 2.0
    for w in surprise_words:
        if w in text: scores["Surprise"] += 2.0

    if "စိတ်ဖိစီး" in text or "ပင်ပန်း" in text:
        scores["Sadness"] += 1.5
        scores["Fear"]    += 1.5
    if "နေမကောင်း" in text or "ဖျား" in text or "နာ" in text:
        scores["Sadness"] += 2.0
        scores["Fear"]    += 0.5
    if "မပြောချင်" in text or "မဖြေချင်" in text:
        scores["Fear"]    += 2.0
        scores["Sadness"] += 0.5

    if sum(scores.values()) == 0:
        for k in scores:
            scores[k] = 1.0

    total = sum(scores.values())
    percentages = {
        label: round((score / total) * 100, 1)
        for label, score in scores.items()
    }

    top_label = max(percentages, key=percentages.get)
    top_conf  = percentages[top_label]

    return {
        "label":      top_label,
        "confidence": top_conf,
        "scores":     percentages
    }

# ─────────────────────────────────────────────
# BOT RESPONSE
# ─────────────────────────────────────────────

# FIX 3: is_quit now uses any() so partial matches work.
# "တာ့တာ နောင်မှတွေ့မယ်" → contains "တာ့တာ" → exits correctly.
def is_quit(text):
    cleaned = normalize_myanmar(text).lower().strip()
    return any(q.lower() in cleaned for q in SCRIPTS["my"]["quits"])

def rule_respond(text):
    text = preprocess_text(text)

    if is_quit(text):
        return random.choice(SCRIPTS["my"]["finals"])

    keywords = sorted(
        SCRIPTS["my"]["keywords"],
        key=lambda x: x[2],
        reverse=True
    )
    for pattern, responses, _ in keywords:
        if re.search(pattern, text):
            return random.choice(responses)

    return "ဆက်ပြောပါ။"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": random.choice(SCRIPTS["my"]["initials"])
        }
    ]

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("🤖 မြန်မာ Elyza")

# ─────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────
chat_html = '<div class="chat-wrap">'

for m in st.session_state.messages:
    role        = m["role"]
    badge       = ""
    details_html = ""

    if "emotion" in m:
        emoji = EMOTION_EMOJI.get(m["emotion"], "")
        conf  = m.get("confidence", None)
        if conf is not None:
            badge = f'<div class="emotion-badge">{emoji} {m["emotion"]} ({conf}%)</div>'
        else:
            badge = f'<div class="emotion-badge">{emoji} {m["emotion"]}</div>'

    if "emotion_scores" in m:
        scores = m["emotion_scores"]
        details_html = (
            f'<div class="small-score">'
            f'Sadness {scores["Sadness"]}% | '
            f'Joy {scores["Joy"]}% | '
            f'Love {scores["Love"]}% | '
            f'Anger {scores["Anger"]}% | '
            f'Fear {scores["Fear"]}% | '
            f'Surprise {scores["Surprise"]}%'
            f'</div>'
        )

    if role == "bot":
        avatar = f'<img class="bot-av" src="{AVATAR_URI}"/>'
    else:
        avatar = '<div class="user-av">🧑</div>'

    text = htmllib.escape(m["text"]).replace("\n", "<br>")

    chat_html += (
        f'<div class="msg-row {role}">'
        f'{avatar}'
        f'<div class="msg-content">'
        f'<div class="bubble {role}">{text}</div>'
        f'{badge}'
        f'{details_html}'
        f'</div>'
        f'</div>'
    )

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────
def handle_user_message(user_input):
    cleaned_input = normalize_myanmar(user_input)

    if not cleaned_input:
        return

    emotion_result = detect_emotion_scores(cleaned_input)

    st.session_state.messages.append({
        "role":           "user",
        "text":           cleaned_input,
        "emotion":        emotion_result["label"],
        "confidence":     emotion_result["confidence"],
        "emotion_scores": emotion_result["scores"]
    })

    response = rule_respond(cleaned_input)

    st.session_state.messages.append({
        "role": "bot",
        "text": response
    })

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])

    user_input = col1.text_input(
        "Message",
        key=f"user_input_{st.session_state.input_key}",
        label_visibility="collapsed",
        placeholder="စာရိုက်ပါ..."
    )

    send = col2.form_submit_button("Send", use_container_width=True)

if send:
    handle_user_message(user_input)
    st.session_state.input_key += 1
    st.rerun()