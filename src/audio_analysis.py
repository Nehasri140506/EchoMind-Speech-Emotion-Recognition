import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. AUDIO FILE
# --------------------------------------------------

audio_file = (
    "dataset/RAVDESS/"
    "Audio_Speech_Actors_01-24/"
    "Actor_01/"
    "03-01-01-01-01-01-01.wav"
)


# --------------------------------------------------
# 2. LOAD AUDIO
# --------------------------------------------------

audio, sample_rate = librosa.load(
    audio_file,
    sr=None
)


# --------------------------------------------------
# 3. BASIC AUDIO INFORMATION
# --------------------------------------------------

duration = librosa.get_duration(
    y=audio,
    sr=sample_rate
)

print("=" * 50)
print("       SPEECH AUDIO ANALYSIS")
print("=" * 50)

print(f"Audio File     : {audio_file}")
print(f"Sampling Rate  : {sample_rate} Hz")
print(f"Duration       : {duration:.2f} seconds")
print(f"Number Samples : {len(audio)}")


# --------------------------------------------------
# 4. ZERO CROSSING RATE
# --------------------------------------------------

zcr = librosa.feature.zero_crossing_rate(
    audio
)

zcr_mean = np.mean(zcr)

print(f"Zero Crossing Rate : {zcr_mean:.4f}")


# --------------------------------------------------
# 5. RMS ENERGY
# --------------------------------------------------

rms = librosa.feature.rms(
    y=audio
)

rms_mean = np.mean(rms)

print(f"RMS Energy         : {rms_mean:.4f}")


# --------------------------------------------------
# 6. MFCC FEATURES
# --------------------------------------------------

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print(f"MFCC Shape         : {mfcc.shape}")


# --------------------------------------------------
# 7. WAVEFORM
# --------------------------------------------------

plt.figure(figsize=(12, 4))

librosa.display.waveshow(
    audio,
    sr=sample_rate
)

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    "visualizations/waveform.png"
)

plt.show()


# --------------------------------------------------
# 8. SPECTROGRAM
# --------------------------------------------------

spectrogram = np.abs(
    librosa.stft(audio)
)

spectrogram_db = librosa.amplitude_to_db(
    spectrogram,
    ref=np.max
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    spectrogram_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.title("Speech Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()

plt.savefig(
    "visualizations/spectrogram.png"
)

plt.show()


# --------------------------------------------------
# 9. MFCC VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mfcc,
    sr=sample_rate,
    x_axis="time"
)

plt.colorbar()

plt.title("MFCC Features")
plt.xlabel("Time (seconds)")
plt.ylabel("MFCC Coefficients")

plt.tight_layout()

plt.savefig(
    "visualizations/mfcc.png"
)

plt.show()
# --------------------------------------------------
# 10. FINISHED
# --------------------------------------------------

print("=" * 50)
print("Audio analysis completed successfully!")
print("Graphs saved in the visualizations folder.")
print("=" * 50)