import pandas as pd
import time
import csv
from google import genai

# 1. Setup
API_KEY = "YOUR API KEY"
client = genai.Client(api_key=API_KEY)

INPUT_FILE = "emotions.csv"   # Should have columns: Text, Label
OUTPUT_FILE = "output.csv" # Will have: Text, Label, Feeling_label
BATCH_SIZE = 40            # Smaller batch is safer for the free tier output limit

# Mapping dictionary
LABEL_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

def translate_and_map(batch_df):
    # Prepare a formatted string for the AI to process
    prompt_data = ""
    for _, row in batch_df.iterrows():
        feeling = LABEL_MAP.get(int(row['label']), "Unknown")
        prompt_data += f"Sentence: {row['text']} | ID: {row['label']} | Feel: {feeling}\n"

    prompt = f"""
    Translate the 'Sentence' part into natural, conversational Burmese.
    Keep the 'ID' and 'Feel' exactly as they are.
    Return ONLY a CSV format with no header, no bold text, and no extra words.
    Format: "Burmese Text", ID, Feel
    
    Data:
    {prompt_data}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# 2. Execution
df = pd.read_csv(INPUT_FILE)

# Initialize Output File with Header
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Text", "Label", "Feeling_label"])

# Loop through the dataframe in batches
for i in range(0, len(df), BATCH_SIZE):
    print(f"Translating rows {i} to {i+BATCH_SIZE}...")
    batch = df.iloc[i : i+BATCH_SIZE]
    
    result = translate_and_map(batch)
    
    if result:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(result + "\n")
    
    # Crucial for Free Tier: Wait to avoid "429 Too Many Requests"
    # The free limit is roughly 15 requests per minute.
    time.sleep(5) 

print(f"Process complete! Saved to {OUTPUT_FILE}")