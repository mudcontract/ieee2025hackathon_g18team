import serial
import mne
from scipy.signal import butter, iirnotch, sosfiltfilt
import numpy as np
import time
from collections import deque

"""
Dataset Information:
    Stimulation frequencies: 9, 10, 12 and 15 Hz (column 1, 2, 3 and 4 in classInfo_4_5.m)
    Sampling rate: 256 Hz
    CH1: sample time
    CH2-9: EEG
    CH10: trigger info (LED on...1, LED off...0);
    CH11: LDA classification output
"""

# 1. Load the EDF file
raw = mne.io.read_raw_edf('your_file.edf', preload=True)  # preload=True loads data into memory
print(raw.info)  # View metadata
# Access signal data and times
data, times = raw[:]

# Plot the raw signals (optional)
raw.plot()

# You can now preprocess: filtering, resampling, etc.
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)  # Band-pass filter
# 2. Set channel types if not correctly detected (optional but common)
# Example: label EEG channels if needed
# raw.set_channel_types({'EEG Fpz-Cz': 'eeg', 'EEG Pz-Oz': 'eeg'})

# 3. Apply bandpass filter (e.g., 1–40 Hz for EEG)
raw.filter(l_freq=1.0, h_freq=40.0)

# 4. Resample the data (e.g., to 256 Hz to reduce data size)
raw.resample(sfreq=256)

# 5. Remove power line noise (e.g., 50 or 60 Hz notch filter)
raw.notch_filter(freqs=50)

# 6. Plot the signals (optional)
raw.plot(n_channels=10, scalings='auto')

# 7. Epoch the data into fixed time segments (e.g., 2-second epochs)
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=0.0, preload=True)
print(epochs)

# 8. (Optional) Run ICA to remove eye-blink or cardiac artifacts
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(epochs)

# Visualize ICA components to manually exclude bad ones
ica.plot_components()

# Mark bad components (you'll manually choose e.g. [0, 1] if they represent eye blinks)
# ica.exclude = [0, 1]  # <-- Example

# Apply ICA cleaning
# ica.apply(epochs)

# 9. Save cleaned data (optional)
# epochs.save('cleaned-epo.fif', overwrite=True)

def band_power(freqs, power, band):
    low, high = band
    idx = np.where((freqs >= low) & (freqs <= high))
    return np.sum(power[idx])

def compute_fft(signal, fs):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)2
    return freqs, power

def eeg_filter_sos(data, fs=250, bandpass=(1, 45), notch=50):
    low = bandpass[0] / (0.5 * fs)
    high = bandpass[1] / (0.5 * fs)
    sos = butter(N=4, Wn=[low, high], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, data)
    w0 = notch / (0.5 * fs)
    b, a = iirnotch(w0, Q=30)
    filtered = sosfiltfilt([[b[0], b[1], b[2], a[0], a[1], a[2]]], filtered)
    return filtered

"""
def preprocessing(channel):
    verf, gain = 4.5, 24
    scale = verf/(gain*(223))
    eeg = [iscale1e6 for i in channel]
    mean = np.mean(eeg)
    return [x-mean for x in eeg]


Preprocessing may include applying filtering functions, removing corrupted or noisy data, and other operations to clean the dataset. The processed results are then saved for further analysis.

Modeling can involve several approaches:
Mathematical modeling, which focuses on analytical or functional computation using mathematical formulas;
Machine learning modeling, which applies classical algorithms such as Support Vector Machines (SVM) or Random Forests;
Deep learning modeling, which employs neural network architectures such as MLP, CNN, MAMBA, or Transformer models.

Visualization plays an important role as well. It may include:
Plotting signal data to demonstrate the effects of preprocessing;
Visualizing the distribution of results (for instance, performance metrics for different categories);
Comparing outcomes under different conditions or model configurations to highlight differences in performance.
"""