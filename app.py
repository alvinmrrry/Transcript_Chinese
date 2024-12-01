import os
import yt_dlp  # YouTube video downloader for audio extraction
import streamlit as st
from moviepy.editor import AudioFileClip  # For audio file processing
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

def split_audio(audio_path, chunk_length_sec=180):  # Default chunk size: 3 minutes (180 seconds)
    """Split audio into chunks of specified length."""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = int(audio_clip.duration)  # Total duration of the audio in seconds
        chunks = []

        # Create subclips of the audio, each of 3 minutes
        for start_time in range(0, duration, chunk_length_sec):
            end_time = min(start_time + chunk_length_sec, duration)
            subclip = audio_clip.subclip(start_time, end_time)
            chunk_filename = f"chunk_{start_time}.mp3"
            subclip.write_audiofile(chunk_filename, codec='mp3')
            chunks.append(chunk_filename)

        audio_clip.close()
        return chunks
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []

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
    """Main function to process video transcript."""
    video_id = extract_video_id(url)
    if not video_id: raise ValueError("Invalid URL.")

    # Download audio in best available format
    audio_path = download_audio(url, video_id)
    if not audio_path: raise ValueError("Audio download failed.")

    # Split the audio into 3-minute chunks
    audio_chunks = split_audio(audio_path)

    all_transcripts = []
    for chunk in audio_chunks:
        # Transcribe each audio chunk
        transcript = transcribe_audio(chunk)
        if transcript:
            all_transcripts.append(transcript)
        # Clean up audio chunk after processing
        os.remove(chunk)

    # Combine all transcripts
    full_transcript = "\n".join(all_transcripts)
    
    # Translate the transcript to Chinese
    translated_text = translate_to_chinese(full_transcript)
    if translated_text:
        # Save the translated transcript to a file
        with open(f"{video_id}_transcript_chinese.txt", "w", encoding="utf-8") as f:
            f.write(translated_text)
        print("Transcript processing completed.")
        return translated_text
    else:
        print("Error in translation.")
    
    # Clean up original audio file
    os.remove(audio_path)

    return None

# Streamlit UI
st.title("YouTube Video Transcript & Translation")
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    try:
        st.write("Processing your video...")

        # Call the processing function
        transcript = process_video_transcript(video_url)

        if transcript:
            # Show the translated transcript in Streamlit
            st.subheader("Chinese Transcript")
            st.text_area("Transcript", transcript, height=400)
        else:
            st.error("Failed to process video.")

    except Exception as e:
        st.error(f"Error processing video: {e}")
