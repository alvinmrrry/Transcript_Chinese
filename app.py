import os
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from pydub import AudioSegment
from groq import Groq

# Initialize Groq client
client = Groq(api_key='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')

# File size limit (25MB)
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes

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

def compress_audio(input_path, output_path):
    """Compress audio to ensure it is under 25MB in size."""
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Lower the audio bitrate or reduce quality to compress the file
        # Save as mp3 with a low bitrate (e.g., 128kbps)
        audio.export(output_path, format="mp3", bitrate="128k")

        # Check the file size, and adjust if necessary
        if os.path.getsize(output_path) > MAX_FILE_SIZE:
            print("File is still too large after compression, reducing further...")
            audio = audio.set_frame_rate(22050)  # Reduce sample rate for further compression
            audio.export(output_path, format="mp3", bitrate="96k")  # Try a lower bitrate

        return output_path
    except Exception as e:
        print(f"Error compressing audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, f.read()), model="whisper-large-v3-turbo")
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

    # Compress audio to ensure it's below the size limit
    compressed_audio_path = f"{video_id}_compressed.mp3"
    compressed_audio_path = compress_audio(audio_path, compressed_audio_path)
    if not compressed_audio_path: raise ValueError("Audio compression failed.")

    # Transcribe the audio using Whisper API
    transcript = transcribe_audio(compressed_audio_path)
    if transcript:
        translated_text = translate_to_chinese(transcript)  # Translate transcript to Chinese
        if translated_text:
            # Save the transcript to a file
            with open(f"{video_id}_transcript_chinese.txt", "w", encoding="utf-8") as f:
                f.write(translated_text)
            print("Transcript processing completed.")
            return translated_text
        else:
            print("Error in translation.")
    else:
        print("Error in transcription.")

    # Clean up audio files
    os.remove(audio_path)
    os.remove(compressed_audio_path)

    return None

# Streamlit UI
st.title("YouTube Video Transcript & Translation with Whisper and Audio Compression")
video_url = st.text_input("Enter YouTube Video URL:")

# Always show "Processing your video..." message
st.write("Processing your video...")

if video_url:
    try:
        # Call the processing function (optional: pass start_time and duration for specific segments)
        transcript = process_video_transcript(video_url)

        if transcript:
            # Show the translated transcript in Streamlit
            st.subheader("Chinese Transcript")
            st.text_area("Transcript", transcript, height=1000)
        else:
            st.error("Failed to process video.")

    except Exception as e:
        st.error(f"Error processing video: {e}")
