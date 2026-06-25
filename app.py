import streamlit as st
from utils import extract_video_id, get_transcript, summarize_transcript

st.set_page_config(page_title="TubeMinds", page_icon="🧠", layout="centered")

st.title("🧠 TubeMinds")
st.caption("Paste a Youtube URL.Get the key ideas in seconds.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Summarize", type="primary"):
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Couldn't find a valid YouTube video ID in that URL.")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript = get_transcript(video_id)
                    st.success(f"Transcript fetched — {len(transcript.split())} words")
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    st.stop()
            with st.spinner("Summarizing with AI....(-15 seconds)"):
                try:
                    summary = summarize_transcript(transcript)
                except Exception as e:
                    st.error(f"Summarization error: {e}")
                    st.stop()
            st.subheader("📋 Summary")
            st.write(summary)

            with st.expander("📄 View raw transcript"):
                st.text_area("Transcript", transcript, height=300)
