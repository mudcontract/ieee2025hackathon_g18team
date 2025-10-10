SSVEP BCI Classification (Team G18)

Analyze an SSVEP EEG dataset and build a full pipeline that covers preprocessing, segmentation, modeling, and evaluation. The final, maintained code lives in the notebook:

- final_code/SSVEP_BCI_Classification_G18.ipynb

The scripts in first_tests/ are exploratory prototypes that did not evolve into the final solution.

## What’s inside

- Data: MATLAB .mat recordings for two subjects in final_code/static/, plus preprocessed .npz datasets in final_code/training_data/.
- Final notebook: end-to-end pipeline with clear sections: processing, sliding windows, neural models, and visualization.
- Models: a lightweight 1D CNN (TinyEEGNet) and a time–frequency two-branch CNN; also a simple SVM baseline.

## Quick start

1) Environment

- Python 3.9+ recommended.
- Install dependencies:

```fish
pip install -r requirements.txt
```

2) Open and run the final notebook

- Open final_code/SSVEP_BCI_Classification_G18.ipynb in VS Code or Jupyter.
- Select a Python kernel with the dependencies installed.
- Execute cells top to bottom; each section explains what it does and what to expect.

3) Data layout (repo paths)

- final_code/static/: raw .mat examples (subject_1/2 sessions)
- final_code/training_data/: pre-generated sliding-window .npz datasets for multiple window sizes

If you add new .mat files, update the corresponding glob/path in the notebook cells.

## Pipeline overview (mirrors the notebook)

1) Data loading & preprocessing

- Create MNE Raw objects with 11 channels: time, 8 EEG channels, trigger, and LDA.
- Apply 1–40 Hz band-pass and 50 Hz notch filters.
- Detect trigger events and assign SSVEP frequencies (9, 10, 12, 15 Hz) following the experimental order.
- Epoch the signal (e.g., 2 s and 8 s). Save epochs to .npz for later modeling.

2) Sliding-window segmentation (detailed, single-subject and batch)

- From longer epochs (e.g., 8 s), generate dense windows (e.g., 2.0 s window with 0.2 s step) to augment data and improve temporal resolution.
- Outputs arrays shaped like (n_windows, n_channels, window_samples) with aligned labels.

3) Neural network classification

- TinyEEGNet (1D CNN): Conv1D → BatchNorm → ReLU → GlobalAvgPool → Linear.
- Time–frequency two-branch CNN: time-domain 1D branch + spectrogram (STFT) 2D branch, fused before classification.
- Training: Adam + CrossEntropyLoss, train/val split with early stopping for the two-branch model.

4) Baselines and visualization

- SVM baseline on flattened, standardized windows (optionally PCA).
- Visualization utilities plot representative epochs by frequency and save figures (e.g., epoch_visualization.png).

## Reproducing results

The notebook contains two TinyEEGNet training runs demonstrating challenges (accuracy near chance) when using a very small/simple model and limited data. The later two-branch time–frequency CNN substantially improves metrics on window lengths ≥ 1.5–2.0 s (see “New version CNN model with double branch” section in the notebook for detailed numbers).

Guidelines to reproduce:

- Use the pre-generated datasets in final_code/training_data/ (e.g., epochs1_sliding_window_subject_1_1.npz, 1 s windows) or generate new ones by running the preprocessing cells.
- Keep channel selection consistent with the notebook (use only EEG channels 1–8 for models).
- Normalize per-epoch per-channel (z-score across time), as in the notebook.

## Adapting to your data

- Replace or add .mat files under final_code/static/.
- Update the file patterns in the data loading cells (glob paths) to your filenames.
- Adjust epoch duration (t_epoch), window length, and step size to trade off latency vs accuracy.

## Tips and known caveats

- Triggers: the notebook maps event order to [15, 12, 10, 9]; confirm your recording protocol and adjust if needed.
- Paths: some early prototype cells used Colab-style paths (/content). The final sections operate on repository files; ensure paths/globs target final_code/static/ or final_code/training_data/ in your local run.
- GPU: PyTorch will use CUDA if available; otherwise it falls back to CPU. Training the two-branch model is faster on GPU.

## Repository structure

- final_code/SSVEP_BCI_Classification_G18.ipynb — Final notebook, use this.
- final_code/static/*.mat — Sample raw recordings used by the notebook.
- final_code/training_data/*.npz — Precomputed sliding-window datasets for quick experiments.
- first_tests/* — Early prototypes (kept for reference only).

## Credits

Participants:
- [Haocheng Wu](https://github.com/TedHaochengWu)
- [Mohammadreza Behbood](https://github.com/mudcontract)
- [Soukaina Hamou](https://github.com/SoukainaHAMOU)
- [Nathan Yu](https://github.com/Littnatenate)
- [Jeronimo Sanchez Santamaria](https://github.com/JeronimoSantamaria)
- Flora Santos
- [Anaya Yorke](https://github.com/anaya33)

Helpful docs: MNE-Python — https://mne.tools/stable/

