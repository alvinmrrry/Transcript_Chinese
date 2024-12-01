import os
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from groq import Groq
from pydub import AudioSegment  # Audio splitting and processing
import tempfile

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

def split_audio(audio_path, chunk_length_seconds=180):  # Default to 3 minutes
    """Split audio into chunks of a given length."""
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(audio_path)
        total_length_ms = len(audio)
        
        # Create chunks based on the duration (convert chunk length to milliseconds)
        chunk_length_ms = chunk_length_seconds * 1000  # Convert to milliseconds
        chunks = []
        
        for i in range(0, total_length_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_file = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
            chunk.export(chunk_file, format="wav")
            chunks.append(chunk_file)
        
        return chunks
    except Exception as e:
        print(f"Error while splitting audio: {e}")
        return []

def transcribe_audio(audio_chunk):
    """Transcribe audio using Whisper."""
    try:
        with open(audio_chunk, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_chunk, f.read()), model="whisper-large-v3-turbo")
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
    """Main function to process video transcript and translate to Chinese."""
    video_id = extract_video_id(url)
    if not video_id: raise ValueError("Invalid URL.")

    # Download audio in best available format
    audio_path = download_audio(url, video_id)
    if not audio_path: raise ValueError("Audio download failed.")
    
    # Check if audio file exists before proceeding
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        return None

    # Split the audio into chunks of 3 minutes each
    audio_chunks = split_audio(audio_path)

    # Store the complete transcript
    full_transcript = ""

    # Process each chunk, transcribe, and translate
    for i, chunk in enumerate(audio_chunks):
        # Transcribe each chunk
        transcript = transcribe_audio(chunk)
        if transcript:
            translated_text = translate_to_chinese(transcript)  # Translate transcript to Chinese
            if translated_text:
                full_transcript += translated_text + "\n"
            else:
                print(f"Error in translation for chunk {i}.")
        else:
            print(f"Error in transcription for chunk {i}.")
        
        # Clean up chunk file
        os.remove(chunk)

    # Clean up audio file
    os.remove(audio_path)

    # Return the complete translated transcript
    return full_transcript

# Streamlit UI
st.title("YouTube Video Transcript & Translation")
video_url = st.text_input("Enter YouTube Video URL:")

# Always show "Processing your video..." message
st.write("Processing your video...")

if video_url:
    try:
        # Call the processing function (this now processes the entire audio)
        transcript = process_video_transcript(video_url)

        if transcript:
            # Show the translated transcript in Streamlit
            st.subheader("Chinese Transcript")
            st.text_area("Transcript", transcript, height=1000)
        else:
            st.error("Failed to process video.")

    except Exception as e:
        st.error(f"Error processing video: {e}")
