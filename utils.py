from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re

load_dotenv()

def extract_video_id(url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id)
    full_text = " ".join([item.text for item in transcript_list])
    return full_text

def summarize_transcript(transcript: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(transcript)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    # MAP: summarize each chunk separately
    partial_summaries = []
    for chunk in chunks:
        response = llm.invoke(
            f"Summarize this part of a video transcript in a few sentences:\n\n{chunk}"
        )
        partial_summaries.append(response.content)

    # REDUCE: combine all partial summaries into one final summary
    combined = "\n".join(partial_summaries)
    final = llm.invoke(
        f"Combine these partial summaries into one clear final summary:\n\n{combined}"
    )
    return final.content