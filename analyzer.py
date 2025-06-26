import re
import pandas as pd
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

emotion_model = pipeline(
    "text-classification", model="nateraw/bert-base-uncased-emotion", top_k=1
)
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")


def load_chat(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    messages = []
    for line in lines:
        line = line.strip()
        # Skip system messages
        if "-" not in line or ":" not in line:
            continue
        # Example: 30/08/2024, 5:30 pm - Sanskar Koserwal: Okay
        match = re.match(
            r"(\d{2}/\d{2}/\d{4}),\s+(\d{1,2}:\d{2})[^\-]*-\s(.+?):\s(.+)", line
        )
        if match:
            date, time, sender, message = match.groups()
            messages.append(
                {
                    "datetime": f"{date} {time}",
                    "sender": sender.strip(),
                    "message": message.strip(),
                }
            )
    return pd.DataFrame(messages)


def analyze_emotions(df):
    def get_label(text):
        res = emotion_model(text)
        # Check if nested list
        if isinstance(res, list) and isinstance(res[0], list):
            return res[0][0]["label"]
        elif isinstance(res, list):
            return res[0]["label"]
        elif isinstance(res, dict):
            return res.get("label", "UNKNOWN")
        else:
            return "UNKNOWN"

    emotions = df["message"].apply(get_label)
    return emotions.value_counts()


def analyze_toxicity(df):
    toxic_count = 0
    for msg in df["message"]:
        result = toxicity_model(msg)[0]
        if result["label"] == "toxic" and result["score"] > 0.8:
            toxic_count += 1
    return toxic_count


def affection_score(df):
    love_words = ["love", "miss", "baby", "jaan", "sweet", "dear", "cutie", "beautiful"]
    count = sum(
        any(word in msg.lower() for word in love_words) for msg in df["message"]
    )
    return round(count / len(df) * 100, 2)


def chat_balance(df):
    counts = df["sender"].value_counts().to_dict()
    senders = list(counts.keys())
    if len(senders) == 2:
        return {
            "Sender A": senders[0],
            "Sender B": senders[1],
            "A Count": counts[senders[0]],
            "B Count": counts[senders[1]],
            "Message Ratio": round(counts[senders[0]] / counts[senders[1]], 2),
        }
    return {}


def generate_wordcloud(df):
    text = " ".join(df["message"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.png")


def run_analysis(chat_path):
    df = load_chat(chat_path)
    if df.empty:
        return None
    emotion_counts = analyze_emotions(df)
    toxic = analyze_toxicity(df)
    affection = affection_score(df)
    balance = chat_balance(df)
    generate_wordcloud(df)
    return {
        "Emotion Counts": emotion_counts,
        "Toxic Messages": toxic,
        "Affection Score": affection,
        "Chat Balance": balance,
    }
