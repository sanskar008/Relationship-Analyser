import streamlit as st
from analyzer import run_analysis

st.title("💌 Relationship Analyzer")

uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type=["txt"])

if uploaded_file:
    with open("uploaded_chat.txt", "wb") as f:
        f.write(uploaded_file.read())

    result = run_analysis("uploaded_chat.txt")

    if result is None:
        st.error("No valid chat data found!")
    else:
        st.subheader("📊 Emotion Distribution")
        st.bar_chart(result["Emotion Counts"])

        st.subheader("🔥 Toxic Messages")
        st.write(f"Total Toxic Messages: {result['Toxic Messages']}")
        for msg in result["Toxic Message Details"]:
            st.markdown(
                f"- **{msg['sender']}** (score: {msg['score']:.2f}): {msg['message']}"
            )

        st.subheader("❤️ Affection Scores (%)")
        st.json(result["Affection Score"])

        st.subheader("⚖️ Chat Balance (%)")
        st.json(result["Chat Balance"])

        st.subheader("📅 Most Active Days")
        st.write(result["Day Time Analysis"]["day_counts"])
        st.write("Per Person:")
        st.write(result["Day Time Analysis"]["day_per_person"])

        st.subheader("⏰ Most Active Hours")
        st.write(result["Day Time Analysis"]["hour_counts"])
        st.write("Per Person:")
        st.write(result["Day Time Analysis"]["hour_per_person"])

        st.subheader("💞 Who is More Loving?")
        st.json(result["Loving Scores"])

        st.subheader("⚡ Effort Scores")
        st.json(result["Effort Scores"])

        st.subheader("💡 Suggestions for Improvement")
        for person, advices in result["Suggestions"].items():
            st.markdown(f"**{person}**:")
            for advice in advices:
                st.write(f"- {advice}")

        st.subheader("🌐 Wordcloud of Chat")
        st.image("wordcloud.png")
