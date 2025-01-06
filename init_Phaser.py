import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
N = 10000  # number of samples to simulate

# Create a tone to act as the transmitter signal
t = np.arange(N) / sample_rate  # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)  # Transmitter signal

d = 0.5  # half wavelength spacing
Nr = 3  # Number of receiver elements
theta_degrees = 20  # Direction of arrival (feel free to change this, it's arbitrary)
theta = theta_degrees / 180 * np.pi  # Convert to radians

# Steering vector (3x1)
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))

s = s.reshape(-1, 1)  # Make s a column vector (3x1)
tx = tx.reshape(1, -1)  # Make tx a row vector (1x10000)

# Simulate the received signal X through matrix multiplication
X = s @ tx  # X will have dimensions (3, 10000)
print(X.shape)  # Should print (3, 10000)

# Add noise to the received signal
n = np.random.randn(Nr, N) + 1j * np.random.randn(Nr, N)  # Complex noise
X = X + 0.2 * n  # Add noise to the received signal

# Plot the real part of the received signal for the first 200 samples
plt.plot(np.asarray(X[0, :]).squeeze().real[0:200])
plt.plot(np.asarray(X[1, :]).squeeze().real[0:200])
plt.plot(np.asarray(X[2, :]).squeeze().real[0:200])
plt.show()

# Conventional delay-and-sum beamformer weights
w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))

# Apply the weights to the received signal (beamforming)
X_weighted = w.conj().T @ X  # Perform beamforming (1x10000)
print(X_weighted.shape)  # Should print (1, 10000)

# Scan over different angles and compute the beamforming output
theta_scan = np.linspace(-np.pi, np.pi, 1000)  # 1000 different thetas between -180 and +180 degrees
results = []

for theta_i in theta_scan:
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))  # Compute weights for the current angle
    X_weighted = w.conj().T @ X  # Apply the weights to the received signal
    results.append(10 * np.log10(np.var(X_weighted)))  # Power in dB

results -= np.max(results)  # Normalize the results (optional)

# Print the angle that gave us the max value
print(theta_scan[np.argmax(results)] * 180 / np.pi)  # Angle in degrees

# Plot the results
plt.plot(theta_scan * 180 / np.pi, results)  # Plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()
plt.show()
