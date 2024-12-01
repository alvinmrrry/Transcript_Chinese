import os
import numpy as np
import audioread
import soundfile as sf
from scipy.signal import resample

def compress_audio(input_file, output_file, target_size_mb=25):
    """Compress audio file to a target size (in MB) using audioread and numpy."""
    try:
        # Read the audio file using audioread
        with audioread.audio_open(input_file) as f:
            sample_rate = f.samplerate
            num_channels = f.channels
            duration = f.duration  # Duration of the audio in seconds

            # Debug: Print audio properties
            print(f"Audio Properties: Sample Rate = {sample_rate}, Channels = {num_channels}, Duration = {duration} seconds")

            total_samples = int(duration * sample_rate)
            audio_data = np.zeros((total_samples, num_channels), dtype=np.float32)

            # Read audio data into numpy array
            for i, buf in enumerate(f):
                audio_data[i * f.block_size:(i + 1) * f.block_size] = np.frombuffer(buf, dtype=np.float32).reshape(-1, num_channels)

        # Debug: Check the size of the audio data
        print(f"Audio Data Shape: {audio_data.shape}")

        # Calculate the target bitrate (in kbps) based on target file size
        target_bitrate = (target_size_mb * 8 * 1024) / duration  # Calculate target bitrate in kbps
        print(f"Calculated Target Bitrate: {target_bitrate:.2f} kbps")

        # Downsample (reduce quality) to achieve the desired bitrate
        target_sample_rate = int(sample_rate * (target_bitrate / 128))  # Simplified calculation for bitrate reduction
        if target_sample_rate < 8000:
            target_sample_rate = 8000  # Don't go below 8000 Hz

        # Resample the audio data
        resampled_audio = resample(audio_data, int(audio_data.shape[0] * target_sample_rate / sample_rate))

        # Save the resampled audio to a new file
        sf.write(output_file, resampled_audio, target_sample_rate)
        
        print(f"File compressed and saved as {output_file}.")
        return output_file
    except Exception as e:
        print(f"Error compressing audio: {e}")
        return None
