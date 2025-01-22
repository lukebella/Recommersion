import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the WAV file
file_path = "happy.wav"  # Replace with your WAV file path
y, sr = librosa.load(file_path, sr=None)

# Plot the raw waveform
plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Raw Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Compute the log mel spectrogram
n_fft = 2048  # Length of FFT window
hop_length = 512  # Number of samples between successive frames
n_mels = 40  # Number of Mel bands

# Compute mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# Convert to log scale (dB)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the log mel spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Log Mel Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency (Hz)")

# Show the plots
plt.tight_layout()
plt.show()
