import os
from groq import Groq
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from pydub import AudioSegment

# Initialize Groq client
client = Groq(api_key='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')

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def download_audio(url, video_id):
    """Download audio from YouTube in a suitable format."""
    try:
        # Download audio directly in best quality available
        ydl_opts = {
            'format': 'bestaudio/best',  # Download the best quality audio
            'outtmpl': f'{video_id}.%(ext)s',  # Save with video ID and the actual file extension
            'postprocessors': [],  # Avoid post-processing for conversion
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
        
        # Find the downloaded audio file with the correct extension
        downloaded_file = f"{video_id}.{info_dict['ext']}"
        return downloaded_file
    except Exception as e:
        print(f"Error: {e}")
        return None

def split_audio(file_path, chunk_length_ms=300000):
    """Split audio into chunks (default: 5 minutes)."""
    try:
        audio = AudioSegment.from_file(file_path)
        return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []

def transcribe_audio(chunk):
    """Transcribe audio chunk using Whisper."""
    try:
        with open(chunk, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(chunk, f.read()), model="whisper-large-v3-turbo")
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def translate_to_chinese(text_chunk):
    """Translate transcript text to Chinese using Groq API."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{
                "role": "system",
                "content": ("You are expert of translating English to Chinese. Please translate the text into Chinese "
                            "and remember not to change the words or sequence of the original sentence.")
            }, {
                "role": "user",
                "content": text_chunk
            }],
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        # Correctly accessing the response attribute
        translated_text = completion.choices[0].message.content
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def process_video_transcript(url):
    """Main function to process video transcript."""
    video_id = extract_video_id(url)
    if not video_id: raise ValueError("Invalid URL.")

    # Download audio in best available format
    audio_path = download_audio(url, video_id)
    if not audio_path: raise ValueError("Audio download failed.")

    # Split the audio into smaller chunks
    chunks = split_audio(audio_path)
    transcript = ""
    
    for i, chunk in enumerate(chunks):
        chunk_file = f"chunk_{i}.m4a"
        chunk.export(chunk_file, format="mp4", codec="aac")  # Export chunk as .m4a or .mp3
        chunk_transcript = transcribe_audio(chunk_file)
        if chunk_transcript:
            translated_text = translate_to_chinese(chunk_transcript)  # Translate transcript to Chinese
            if translated_text:
                transcript += translated_text + "\n"
            else:
                print(f"Failed to translate chunk {chunk_file}.")
        os.remove(chunk_file)  # Clean up chunk after processing
    
    # Write the full transcript to a file
    with open(f"{video_id}_transcript_chinese.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    os.remove(audio_path)  # Clean up original audio file
    print("Transcript processing completed.")

    return transcript

# Streamlit UI
st.title("YouTube Video Transcript & Translation")
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    try:
        st.write("Processing your video...")

        # Call the processing function
        transcript = process_video_transcript(video_url)

        # Show the translated transcript in Streamlit
        st.subheader("Chinese Transcript")
        st.text_area("Transcript", transcript, height=400)
    except Exception as e:
        st.error(f"Error processing video: {e}")
