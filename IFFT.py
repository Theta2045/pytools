import numpy as np
import soundfile as sf
from scipy.io import wavfile


sr, audio = wavfile.read("input.WAV")


audio = audio.astype(np.float32)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

real = audio[0::2]
imag = audio[1::2]
freq_bins = real + 1j * imag


time_signal = np.fft.ifft(freq_bins)


time_signal = time_signal / np.max(np.abs(time_signal))


sf.write("output.wav", time_signal.real.astype(np.float32), sr)
