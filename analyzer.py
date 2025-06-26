import pandas as pd
import nltk
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download punkt for sentence tokenization if needed
nltk.download("punkt")

# Initialize models once
emotion_model = pipeline(
    "text-classification", model="bhadresh-savani/bert-base-uncased-emotion"
)
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")


def load_chat(chat_path):
    """Parse WhatsApp exported chat txt to DataFrame with datetime, sender, message."""
    data = []
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            if " - " in line and ": " in line:
                # Format: "30/08/2024, 3:42 pm - Sender: Message"
                date_part, rest = line.split(" - ", 1)
                sender_part, message = rest.split(": ", 1)
                try:
                    dt = pd.to_datetime(date_part.strip(), dayfirst=True)
                except Exception:
                    continue
                data.append(
                    {
                        "datetime": dt,
                        "sender": sender_part.strip(),
                        "message": message.strip(),
                    }
                )
    df = pd.DataFrame(data)
    return df


def analyze_emotions(df):
    def get_label(text):
        res = emotion_model(text)
        # Handle nested list e.g. [[{'label': 'joy', ...}]]
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


def mark_toxic_messages(df):
    toxic_msgs = []
    for idx, row in df.iterrows():
        result = toxicity_model(row["message"])
        if isinstance(result, list):
            label = result[0]["label"]
            score = result[0]["score"]
        else:
            label = result["label"]
            score = result["score"]

        if label == "toxic" and score > 0.8:
            toxic_msgs.append(
                {
                    "index": idx,
                    "sender": row["sender"],
                    "message": row["message"],
                    "score": score,
                }
            )
    return toxic_msgs


def affection_score(df):
    love_words = ["love", "miss", "baby", "jaan", "sweet", "dear", "cutie", "beautiful"]
    senders = df["sender"].unique()
    scores = {}
    for sender in senders:
        messages = df[df["sender"] == sender]["message"].str.lower()
        affection_count = sum(
            messages.apply(lambda msg: any(word in msg for word in love_words))
        )
        total_msgs = len(messages)
        scores[sender] = (
            round((affection_count / total_msgs) * 100, 2) if total_msgs > 0 else 0
        )
    return scores


def chat_balance(df):
    counts = df["sender"].value_counts(normalize=True) * 100
    return counts.to_dict()


def day_time_analysis(df):
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour_of_day"] = df["datetime"].dt.hour

    day_counts = df["day_of_week"].value_counts()
    day_per_person = df.groupby("sender")["day_of_week"].value_counts()
    hour_counts = df["hour_of_day"].value_counts()
    hour_per_person = df.groupby("sender")["hour_of_day"].value_counts()

    return {
        "day_counts": day_counts,
        "day_per_person": day_per_person,
        "hour_counts": hour_counts,
        "hour_per_person": hour_per_person,
    }


def effort_score(df):
    df["date_only"] = df["datetime"].dt.date
    senders = df["sender"].unique()

    total_msgs = df["sender"].value_counts()
    first_msgs = df.sort_values("datetime").groupby("date_only").first()
    first_msg_senders = first_msgs["sender"].value_counts()

    scores = {}
    for sender in senders:
        msg_count = total_msgs.get(sender, 0)
        first_count = first_msg_senders.get(sender, 0)
        scores[sender] = {
            "messages_sent": msg_count,
            "first_msg_count": first_count,
            "effort_score": msg_count + first_count * 2,
        }
    return scores


def improvement_suggestions(df, toxic_msgs, loving_scores, effort_scores):
    suggestions = {}
    for sender in df["sender"].unique():
        suggestions[sender] = []
        toxic_count = sum(1 for msg in toxic_msgs if msg["sender"] == sender)
        if toxic_count > 2:
            suggestions[sender].append(
                "Try to reduce negative or toxic messages for better communication."
            )
        if loving_scores.get(sender, 0) < 5:
            suggestions[sender].append("Express affection and appreciation more often.")
        effort = effort_scores.get(sender, {})
        if effort.get("effort_score", 0) < 10:
            suggestions[sender].append("Participate more actively in conversations.")
        if not suggestions[sender]:
            suggestions[sender].append("Keep up the good communication!")
    return suggestions


def generate_wordcloud(df):
    text = " ".join(df["message"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    wordcloud.to_file("wordcloud.png")


def run_analysis(chat_path):
    df = load_chat(chat_path)
    if df.empty:
        return None

    emotion_counts = analyze_emotions(df)
    toxic_msgs = mark_toxic_messages(df)
    affection = affection_score(df)
    balance = chat_balance(df)
    day_time = day_time_analysis(df)
    loving_scores = affection  # reuse affection_score
    effort_scores = effort_score(df)
    suggestions = improvement_suggestions(df, toxic_msgs, loving_scores, effort_scores)
    generate_wordcloud(df)

    return {
        "Emotion Counts": emotion_counts,
        "Toxic Messages": len(toxic_msgs),
        "Toxic Message Details": toxic_msgs,
        "Affection Score": affection,
        "Chat Balance": balance,
        "Day Time Analysis": day_time,
        "Loving Scores": loving_scores,
        "Effort Scores": effort_scores,
        "Suggestions": suggestions,
    }
