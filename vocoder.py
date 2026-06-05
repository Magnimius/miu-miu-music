import numpy as np
import librosa
import soundfile
import matplotlib.pyplot as plt
print("Getting Sawtooth")
sr = 22050
duration = 50.0
t = np.linspace(0, duration, int(sr * duration))
carrier = 2 * (t * 220 % 1) - 1  # Sawtooth formula
print("Got Sawtooth")

print("Loading part4u")
modulator, sr = librosa.load('./part4u.wav', sr=None)
print("Got part4u")
min_len = min(len(carrier), len(modulator))
carrier = carrier[:min_len]
modulator = modulator[:min_len]

def frame_signal(signal, frame_size, hop_size):
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.lib.stride_tricks.as_strided(
        signal, shape=(num_frames, frame_size),
        strides=(signal.strides[0] * hop_size, signal.strides[0])
    )
    return frames

frame_size = 1024
hop_size = 512

carrier_frames = frame_signal(carrier, frame_size, hop_size)
modulator_frames = frame_signal(modulator, frame_size, hop_size)

def stft(frames, n_fft):
    return np.fft.rfft(frames, n=n_fft)

n_fft = 1024

carrier_stft = stft(carrier_frames, n_fft)
modulator_stft = stft(modulator_frames, n_fft)

modulator_amplitude = np.abs(modulator_stft)

modulated_stft = carrier_stft * (modulator_amplitude / np.abs(carrier_stft))

def istft(stft_matrix, hop_size):
    num_frames, n_fft = stft_matrix.shape
    frame_size = (n_fft - 1) * 2
    signal = np.zeros(num_frames * hop_size + frame_size - hop_size)
    for n, i in enumerate(range(0, len(signal) - frame_size, hop_size)):
        signal[i:i + frame_size] += np.fft.irfft(stft_matrix[n])
    return signal

output_signal = istft(modulated_stft, hop_size)

output_signal = output_signal / np.max(np.abs(output_signal))
soundfile.write('./output.wav', output_signal, sr)
