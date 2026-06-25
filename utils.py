from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
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

def get_transcript(video_id:str) -> str:
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id)
    full_text = " ".join([item.text for item in transcript_list])
    return full_text

def summarize_transcript(transcript: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]

    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        api_key = os.getenv("GROQ_API_KEY"),
        temperature = 0.3
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.invoke(docs)
    return result["output_text"]
