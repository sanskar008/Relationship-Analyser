import streamlit as st
from analyzer import run_analysis

st.title("❤️ Hinglish Relationship Analyzer (Free Tool)")

uploaded_file = st.file_uploader("Upload chat file (.txt)", type="txt")

if uploaded_file:
    with open("uploaded_chat.txt", "wb") as f:
        f.write(uploaded_file.read())

    result = run_analysis("uploaded_chat.txt")

    st.subheader("📊 Emotion Distribution")
    st.bar_chart(result["Emotion Counts"])

    st.subheader("🔥 Toxicity")
    st.write(f"Toxic Messages: {result['Toxic Messages']}")

    st.subheader("💖 Affection Score")
    st.write(f"Affection Score: {result['Affection Score']}%")

    st.subheader("⚖️ Chat Balance")
    st.json(result["Chat Balance"])

    st.subheader("🧠 Common Topics")
    st.image("wordcloud.png", caption="WordCloud of Chat Messages")
