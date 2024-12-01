import os
import yt_dlp  # YouTube video downloader for audio extraction
import requests  # For making API requests
import streamlit as st

# Hugging Face API URL and headers for authentication
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": "Bearer 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"}  # Replace with your Hugging Face API key

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def download_audio(url, video_id):
    """Download audio from YouTube in a suitable format."""
    try:
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

def query(filename):
    """Send audio file to Hugging Face Whisper API for transcription."""
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    
    if response.status_code == 200:
        return response.json()  # If successful, return the transcription result.
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def process_video_transcript(url):
    """Main function to process video transcript."""
    video_id = extract_video_id(url)
    if not video_id: 
        raise ValueError("Invalid URL.")

    # Download audio in best available format
    audio_path = download_audio(url, video_id)
    if not audio_path: 
        raise ValueError("Audio download failed.")

    # Transcribe the audio using Hugging Face Whisper API
    transcript_result = query(audio_path)
    if transcript_result and 'text' in transcript_result:
        transcript = transcript_result['text']
        print("Transcription completed.")
        return transcript
    else:
        print("Error in transcription.")
        return None

    # Clean up audio file
    os.remove(audio_path)

# Streamlit UI
st.title("YouTube Video Transcript & Translation with Hugging Face Whisper")
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    try:
        st.write("Processing your video...")

        # Call the processing function
        transcript = process_video_transcript(video_url)

        if transcript:
            # Show the transcript in Streamlit
            st.subheader("Transcript")
            st.text_area("Transcript", transcript, height=1000)
        else:
            st.error("Failed to process video.")

    except Exception as e:
        st.error(f"Error processing video: {e}")
