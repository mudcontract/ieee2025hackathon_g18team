import mne
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 1. Load EDF file
raw = mne.io.read_raw_edf(
    r".EDF file path",
    preload=True
)
ch_names = ['time'] + [f'EEG{i}' for i in range(1, 9)] + ['trigger', 'LDA']
raw.rename_channels(dict(zip(raw.ch_names, ch_names)))
channel_types = ['misc'] + ['eeg'] * 8 + ['stim', 'misc']
raw.set_channel_types(dict(zip(ch_names, channel_types)))

# preprocessing
raw.filter(l_freq=1.0, h_freq=40.0)
raw.notch_filter(freqs=50)

#to see the raw data (optional)
raw.plot(
    scalings='auto',
    title='Raw EEG (All Channels)',
    show=True,
    block=True              #stop untill closing window
)

# 2. Load class_info.m one-hot label file
class_info = np.loadtxt(
    r"classInfo_4_5.m"
)
labels = np.argmax(class_info, axis=1)  # change to  0,1,2,3
freq_map = {0: 15, 1: 12, 2: 10, 3: 9}

# 3. Extract events from trigger channel
events = mne.find_events(raw, stim_channel='trigger', shortest_event=1)

print(f"Found {len(events)} trigger events.")

n_events = min(len(events), len(labels))
print("same??",len(events)==len(labels))
events = events[:n_events]
labels = labels[:n_events]

# 4. Assign event IDs based on frequency
event_id = {'9Hz': 9, '10Hz': 10, '12Hz': 12, '15Hz': 15}
# replace events found with actual freq
for i in range(n_events):
    freq = freq_map[labels[i]]
    events[i, 2] = freq

# 5. Segment into epochs based on triggers
epochs = mne.Epochs(
    raw, events, event_id=event_id,
    tmin=0.0, tmax=2.0, baseline=None, preload=True
)
print(epochs)

data = epochs.get_data()            # shape: (n_epochs, n_channels, n_times)
labels = epochs.events[:, -1]
# save as npz
np.savez(
    r"C:\Users\wu\Desktop\code\python\eeg\hackthon\epochs_dataset.npz",
    data=data,
    labels=labels
)

print("Saved EEG dataset to npz file")
print("data shape:", data.shape)
print("labels shape:", labels.shape)

# 6. Optional: Separate “triggered” and “non-triggered”
triggered_data = epochs.get_data()        # shape: (n_epochs, n_channels, n_times)
untriggered_raw = raw.copy().crop(0, events[0, 0] / raw.info['sfreq'])

print("Triggered EEG shape:", triggered_data.shape)
print("Untriggered duration (s):", untriggered_raw.times[-1])

