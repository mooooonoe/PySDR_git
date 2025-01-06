import numpy as np
import matplotlib.pyplot as plt

# Generate the time-domain signal
t = np.arange(100)
s = np.sin(0.15 * 2 * np.pi * t)

# Compute the FFT
S = np.fft.fft(s)
N = len(S)  # Number of samples
freq = np.fft.fftfreq(N)  # Frequency axis

# Calculate magnitude and phase
S_mag = np.abs(S)
S_phase = np.angle(S)

# Only take the first half of the spectrum for real signals
half_N = N // 2
freq = freq[:half_N]
S_mag = S_mag[:half_N]
S_phase = S_phase[:half_N]

# Plot magnitude and phase
plt.figure(figsize=(12, 6))

# Magnitude plot
plt.subplot(2, 1, 1)
plt.plot(freq, S_mag, '.-')
plt.title("Magnitude Spectrum")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid()

# Phase plot
plt.subplot(2, 1, 2)
plt.plot(freq, S_phase, '.-')
plt.title("Phase Spectrum")
plt.xlabel("Normalized Frequency")
plt.ylabel("Phase (radians)")
plt.grid()

plt.tight_layout()
plt.show()
