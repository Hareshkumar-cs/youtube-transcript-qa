import streamlit as st
from ai_youtube import run_qa   


st.title("🎬 AI YouTube Video Question Answering System")
st.caption("Ask questions from any YouTube video using AI and transcripts")


video_id_input = st.text_input("Enter YouTube Video ID", value="hEgO047GxaQ")
question_input = st.text_input("Ask a question about the video")


if st.button("Run QA"):
    if not video_id_input or not question_input:
        st.warning("Please enter both Video ID and Question")
    else:
        with st.spinner("Processing..."):
            result = run_qa(video_id_input, question_input)

        st.success("Answer:")
        st.write(result)
