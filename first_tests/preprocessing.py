import numpy as np
from scipy.io import loadmat
import mne

# =========================
# 1. Load EEG data from .mat
# =========================
mat_file = r"C:\Users\wu\Desktop\code\python\eeg\hackthon\ssvep\subject_1_fvep_led_training_1.mat"
data = loadmat(mat_file)

eeg_data = data['y']  # shape: (11, n_times)
sfreq = float(data['fs'].squeeze())  # sampling rate

print(f"EEG data shape: {eeg_data.shape}, Sampling rate: {sfreq} Hz")

# =========================
# 2. Create MNE Raw object
# =========================
# Channel names
ch_names = ['time'] + [f'EEG{i}' for i in range(1, 9)] + ['trigger', 'LDA']

# Create info structure
info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=['misc'] + ['eeg']*8 + ['stim', 'misc']  # misc for non-EEG, stim for trigger
)

# Create RawArray
raw = mne.io.RawArray(eeg_data, info)

# Quick visualization
#raw.plot(duration=10, n_channels=11, scalings='auto', block=True)

# =========================
# 3. Preprocessing
# =========================
raw.filter(l_freq=1.0, h_freq=40.0)  # bandpass 1-40 Hz
raw.notch_filter(freqs=50)           # notch 50 Hz

# Optional: visualize preprocessed signal
#raw.plot(scalings='auto', title='Preprocessed EEG', block=True)

# =========================
# 4. Generate labels from trigger sequence
# =========================
events = mne.find_events(raw, stim_channel='trigger', shortest_event=1)
print(f"Found {len(events)} trigger events")

stim_order = [15, 12, 10, 9]  # 循环顺序
n_events = len(events)
labels = np.array([stim_order[i % len(stim_order)] for i in range(n_events)])
print("First 10 labels:", labels[:10])

# 5. Assign event IDs based on frequency
event_id = {'9Hz': 9, '10Hz': 10, '12Hz': 12, '15Hz': 15}

# Replace events found with actual frequency label
for i in range(n_events):
    events[i, 2] = labels[i]

# =========================
# 6. Create epochs based on trigger
# =========================
event_id = {'9Hz': 9, '10Hz': 10, '12Hz': 12, '15Hz': 15}
epochs = mne.Epochs(
    raw, events, event_id=event_id,
    tmin=0.0, tmax=2.0, baseline=None, preload=True
)
print(epochs)

# Extract data and labels for ML
data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
labels = epochs.events[:, -1]

# Save as npz for deep learning
np.savez(
    r"C:\Users\wu\Desktop\code\python\eeg\hackthon\epochs_dataset.npz",
    data=data,
    labels=labels
)
print(" Saved EEG dataset to npz file")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# =========================
# 7. Optional: separate triggered vs untriggered segments
# =========================
triggered_data = data  # same as epochs.get_data()
untriggered_raw = raw.copy().crop(0, events[0, 0] / raw.info['sfreq'])
print("Triggered EEG shape:", triggered_data.shape)
print("Untriggered duration (s):", untriggered_raw.times[-1])
