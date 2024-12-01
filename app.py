import os
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from moviepy.editor import AudioFileClip
from groq import Groq

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

def segment_audio(audio_path, segment_length=180):
    """Segment the audio into parts of segment_length (in seconds)."""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration  # Duration of the full audio in seconds
        segments = []

        # Split the audio into segments of specified length
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)
            segment = audio_clip.subclip(start_time, end_time)
            segment_path = f"segment_{start_time}_{end_time}.mp3"
            segment.write_audiofile(segment_path)
            segments.append(segment_path)
        
        return segments
    except Exception as e:
        print(f"Error during audio segmentation: {e}")
        return []

def process_video_transcript(url):
    """Main function to process video transcript and translate to Chinese."""
    video_id = extract_video_id(url)
    if not video_id: raise ValueError("Invalid URL.")

    # Download audio in best available format
    audio_path = download_audio(url, video_id)
    if not audio_path: raise ValueError("Audio download failed.")

    try:
        # Segment the audio file into 3-minute parts (or a custom duration)
        segments = segment_audio(audio_path, segment_length=180)  # Segment into 3-minute chunks
        full_transcript = []

        for segment in segments:
            # Transcribe each segment
            transcript = transcribe_audio(segment)
            if transcript:
                full_transcript.append(transcript)
            else:
                print(f"Error transcribing segment {segment}")

        # Join all segments into one transcript
        full_transcript_text = "\n".join(full_transcript)
        
        # Translate the transcript to Chinese
        translated_text = translate_to_chinese(full_transcript_text)
        if translated_text:
            # Save the translated transcript to a file
            with open(f"{video_id}_transcript_chinese.txt", "w", encoding="utf-8") as f:
                f.write(translated_text)
            print("Transcript processing completed.")
            return translated_text
        else:
            print("Error in translation.")
        
    except Exception as e:
        print(f"Error processing video segment: {e}")
    finally:
        # Clean up audio files and segments
        os.remove(audio_path)
        for segment in segments:
            os.remove(segment)

    return None

# Streamlit UI
st.title("YouTube Video Transcript & Translation")
video_url = st.text_input("Enter YouTube Video URL:")

# Always show "Processing your video..." message
st.write("Processing your video...")

if video_url:
    try:
        # Call the processing function
        transcript = process_video_transcript(video_url)

        if transcript:
            # Show the translated transcript in Streamlit
            st.subheader("Chinese Transcript")
            st.text_area("Transcript", transcript, height=1000)
        else:
            st.error("Failed to process video.")

    except Exception as e:
        st.error(f"Error processing video: {e}")
