import os
import re
import yt_dlp
from pydub import AudioSegment
from groq import Groq
import streamlit as st

# Initialize Groq client
client = Groq(api_key='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')

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def download_audio(url, video_id):
    """Download audio from YouTube."""
    try:
        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': f'{video_id}.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        return f"{video_id}.webm"
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def convert_to_m4a(input_path):
    """Convert audio to .m4a if needed."""
    audio = AudioSegment.from_file(input_path)
    m4a_path = f"{input_path.split('.')[0]}.m4a"
    audio.export(m4a_path, format="mp4", codec="aac")
    return m4a_path

def split_audio(file_path, chunk_length_ms=300000):
    """Split audio into 5-minute chunks."""
    audio = AudioSegment.from_file(file_path)
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

def transcribe_audio(chunk):
    """Transcribe audio chunk using Whisper."""
    with open(chunk, "rb") as f:
        return client.audio.transcriptions.create(file=(chunk, f.read()), model="whisper-large-v3-turbo").text

def translate_to_chinese(text_chunk):
    """Translate transcript text to Chinese using Groq API."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": ("You are an expert in translating English to Chinese. Please translate the text into Chinese and remember not to change the words or sequence of the original sentence.")
                },
                {"role": "user", "content": text_chunk}
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        # Correctly accessing the response attribute
        translated_text = completion.choices[0].message.content
        return translated_text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return None

def process_video_transcript(url):
    """Main function to process video transcript."""
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid URL.")
        return

    audio_path = download_audio(url, video_id)
    if not audio_path:
        st.error("Audio download failed.")
        return

    if audio_path.endswith(".webm"):
        audio_path = convert_to_m4a(audio_path)

    chunks = split_audio(audio_path)
    transcript = ""
    
    for i, chunk in enumerate(chunks):
        chunk_file = f"chunk_{i}.m4a"
        chunk.export(chunk_file, format="mp4", codec="aac")
        chunk_transcript = transcribe_audio(chunk_file)
        if chunk_transcript:
            translated_text = translate_to_chinese(chunk_transcript)  # Translate transcript to Chinese
            if translated_text:
                transcript += translated_text + "\n"
            else:
                st.warning(f"Failed to translate chunk {chunk_file}.")
        os.remove(chunk_file)  # Clean up chunk after processing
    
    # Save transcript in a text file
    with open(f"{video_id}_transcript_chinese.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    os.remove(audio_path)  # Clean up original audio file
    st.success("Transcript processing completed.")
    return transcript

# Streamlit UI
st.title("YouTube Video Translator & Transcription")

video_url = st.text_input("Enter YouTube Video URL", "")

if video_url:
    st.video(video_url)  # Show video on the Streamlit interface
    
    # Process and display the transcript in Chinese
    transcript = process_video_transcript(video_url)
    
    if transcript:
        st.subheader("Chinese Transcript:")
        st.text_area("Transcript", transcript, height=300)


