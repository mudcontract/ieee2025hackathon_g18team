IEEE 2025 team G18

**Analyze an SSVEP BCI data-set from a healthy person in order to optimize pre-processing, feature extraction and classification algorithms. Compare your results with state-of-the-art algorithms.**

Steps from Ted:

1. Here we have eight EEG channels, and we can map them to their corresponding labels using the trigger information — for example, if the trigger shows that 10 Hz was activated at the first second.
2. Next, I’d do some basic preprocessing, such as band-pass filtering, notch filtering (to remove power-line noise), etc.u might have their own preferred techniques here.
3. After that, segment the data so that each EEG segment corresponds to one stimulation frequency. These segments will serve as the basic modeling units.
4. Then comes the modeling and validation part. My initial idea for the AI model is:
   1. Generate frequency-domain features using Fourier transform or wavelet transform;
   2. Combine them with spatial-temporal features and feed them into a neural network for modeling.
   3. Depending on the trade-off between speed and accuracy, we can also explore other approaches — this is just a direction I’m more familiar with, not necessarily the best one here.
5. Finally, we can evaluate the model and visualize the results.
   The evaluation goals might include accuracy, runtime efficiency, and response time, among others.
6. To push things further, I think we could explore questions like:
   1. Can we achieve good performance using fewer channels?
   2. Can we train with fewer samples?
   3. How does channel combination affect the overall accuracy?

Assignments: 

1. Preprocessing
   1. Ella
   2. Ted
2. Modeling
   1. Ella
   2. Ted
3. Visualization
   1. Jeronimo
   2. Nate Yu

Please read the MNE documentation below for your project assignment.

[https://mne.tools/stable/index.html](https://mne.tools/stable/index.html "PLEASE READ THE MNE LIBRARY")

![1759665990667](image/readme/1759665990667.png)

import mne

1. Load the EDF file

`raw = mne.io.read_raw_edf('your_file.edf', preload=True)  # preload=True loads data into memory
print(raw.info)  # View metadata`

# 2. Set channel types if not correctly detected (optional but common)

# Example: label EEG channels if needed

`raw.set_channel_types()`

# 3. Apply bandpass filter (e.g., 1–40 Hz for EEG)

`raw.filter(l_freq=1.0, h_freq=40.0)`

# 4. Resample the data (e.g., to 256 Hz to reduce data size)

`raw.resample(sfreq=256)`

# 5. Remove power line noise (e.g., 50 or 60 Hz notch filter)

`raw.notch_filter(freqs=50)`

# 6. Plot the signals (optional)

raw.plot(n_channels=10, scalings='auto')

# 7. Epoch the data into fixed time segments (e.g., 2-second epochs)

epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=0.0, preload=True)
print(epochs)

# 8. (Optional) Run ICA to remove eye-blink or cardiac artifacts

ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(epochs)

###### Visualize ICA components to manually exclude bad ones

ica.plot_components()

###### Mark bad components (you'll manually choose e.g. [0, 1] if they represent eye blinks)

ica.exclude = [0, 1]  # <-- Example

Apply ICA cleaning

ica.apply(epochs)

-#9. Save cleaned data (optional)

-#epochs.save('cleaned-epo.fif', overwrite=True)
