import numpy as np
import matplotlib.pyplot as plt
import os, h5py

# Setup and File Path
folder_name = 'information'
file_name = 'subject_1_fvep_led_training_1.mat'
full_path = os.path.join(folder_name, file_name) 

# Load and Extract Data using h5py
if not os.path.exists(full_path):
    print(f"Error: The file was not found at {full_path}")
else:
    print(f"Loading file using h5py from: {full_path}")
    
    with h5py.File(full_path, 'r') as f:
        eeg_data = f['y'][:]

    # Transpose the data
    eeg_data = eeg_data.T

    # Prepare Data for Plotting
    sampling_rate = 256
    eeg_channel = eeg_data[:, 1]
    trigger_channel = eeg_data[:, 9]
    num_samples = eeg_channel.shape[0]
    time_seconds = np.arange(num_samples) / sampling_rate

    # Plot the Signals
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

    axes[0].plot(time_seconds, eeg_channel)
    axes[0].set_title('Raw EEG Signal (One Channel)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].grid(True)

    axes[1].plot(time_seconds, trigger_channel)
    axes[1].set_title('Trigger Channel (Stimulus On/Off)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Status (1=On, 0=Off)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()