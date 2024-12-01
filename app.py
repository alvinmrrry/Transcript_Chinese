import os
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from groq import Groq

# Initialize Groq client
client = Groq(api_key='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')

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def download_audio(url, video_id):
    """Download audio from YouTube in normal quality."""
    try:
        # Specify a lower quality audio format by using 'bestaudio' with lower audio quality settings
        ydl_opts = {
            'format': 'bestaudio[ext=mp3]',  # Choose m4a or mp3, lower quality audio formats
            'outtmpl': f'{video_id}.%(ext)s',  # Save with video ID and the actual file extension
            'postprocessors': [],  # Avoid post-processing for conversion
            'noplaylist': True,  # Avoid downloading playlists
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
        
        # Find the downloaded audio file with the correct extension
        downloaded_file = f"{video_id}.{info_dict['ext']}"
        return downloaded_file
    except Exception as e:
        print(f"Error: {e}")
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
        
        # Correctly accessing the response attribute
        translated_text = completion.choices[0].message.content
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def process_video_transcript(url, start_time=0, duration=300):
    """Main function to process video transcript and translate to Chinese."""
    video_id = extract_video_id(url)
    if not video_id: 
        raise ValueError("Invalid URL.")

    # Download audio in normal quality (not best)
    audio_path = download_audio(url, video_id)
    if not audio_path: 
        raise ValueError("Audio download failed.")

    # Only transcribe a portion of the video (based on start_time and duration)
    # You can modify this if necessary to extract specific segments.
    try:
        # Limit the audio to the desired segment using yt-dlp or other tools
        # For now, we are transcribing the entire audio. In practice, you can trim or split the audio.

        transcript = transcribe_audio(audio_path)  # Transcribe the full audio or segment
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
    except Exception as e:
        print(f"Error processing video segment: {e}")
    finally:
        # Clean up audio file
        os.remove(audio_path)

    return None

# Streamlit UI
st.title("YouTube Video Transcript & Translation")
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
