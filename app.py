import os
import yt_dlp  # YouTube video downloader for audio extraction
import ffmpeg  # To manipulate and compress audio files
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
    """Download audio from YouTube in lower quality."""
    try:
        # Download audio with yt-dlp, use lower quality settings
        ydl_opts = {
            'format': 'bestaudio[abr<=128]/mp4',  # Choose audio with bitrate <= 128 kbps, or mp4 audio
            'outtmpl': f'{video_id}.%(ext)s',  # Save with video ID and the actual file extension
            'postprocessors': [],  # Avoid post-processing for conversion
            'noplaylist': True,  # Avoid downloading playlists
            'extractaudio': True,  # Ensure we only extract audio
            'audioquality': 1,  # 1 = best quality, 9 = worst quality; 9 is typically lower quality
            'restrictfilenames': True,  # Prevent unusual characters in filenames
            'logtostderr': False,  # Disable logging to stderr (optional)
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)

        # Find the downloaded file
        downloaded_file = f"{video_id}.{info_dict['ext']}"
        return downloaded_file
    except Exception as e:
        print(f"Error: {e}")
        return None

def compress_audio(input_file, output_file, target_size_mb=25):
    """Compress audio file to a target size (in MB)."""
    try:
        # Estimate target bitrate based on desired size (in MB) and file duration
        duration = ffmpeg.probe(input_file, v='error', select_streams='a:0', show_entries='format=duration')['format']['duration']
        target_bitrate = (target_size_mb * 8 * 1024) / duration  # Calculate target bitrate in kbps

        # Ensure the bitrate doesn't exceed 128kbps, as we want to avoid excessive quality loss
        target_bitrate = min(target_bitrate, 128)

        # Compress the audio using ffmpeg with the target bitrate
        ffmpeg.input(input_file).output(output_file, audio_bitrate=f'{int(target_bitrate)}k').run()

        # Return the compressed audio file path
        return output_file
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
        
        # Correctly accessing the response attribute
        translated_text = completion.choices[0].message.content
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def process_video_transcript(url, start_time=0, duration=300):
    """Main function to process video transcript and translate to Chinese."""
    video_id = extract_video_id(url)
    if not video_id: raise ValueError("Invalid URL.")

    # Download audio in best available format (normal quality)
    audio_path = download_audio(url, video_id)
    if not audio_path: raise ValueError("Audio download failed.")

    # Compress the audio to ensure it's under 25MB
    compressed_audio_path = compress_audio(audio_path, f"{video_id}_compressed.mp3", target_size_mb=25)
    if not compressed_audio_path: raise ValueError("Audio compression failed.")

    # Transcribe the compressed audio to text
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
