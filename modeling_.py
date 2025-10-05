import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# === 1. 加载 npz 文件 ===
data = np.load(r'C:\Users\wu\Desktop\code\python\eeg\hackthon\epochs_dataset.npz')
X = data['data'][:, 1:9, :]  # [N, C, T]
y = data['labels']  # [N]

unique_labels = sorted(set(y),reverse=True)
label_map = {freq: idx for idx, freq in enumerate(unique_labels)}
y = np.array([label_map[val] for val in y])

print(X.shape, y.shape)

print("Unique labels in y:", np.unique(y))
print("Label counts:", np.bincount(y))

print("Mean:", X.mean(), "Std:", X.std())
print("First sample shape:", X[0].shape)
print("Sample 0 mean per channel:", X[0].mean(axis=1))


import numpy as np
import matplotlib.pyplot as plt

fs = 256
sample_idx = 4
signal = X[sample_idx]  # shape: (8, 513)
label = y[sample_idx]

#compute fft
N = signal.shape[1]
freqs = np.fft.rfftfreq(N, d=1/fs)
fft_vals = np.fft.rfft(signal, axis=1)
power = np.abs(fft_vals) ** 2


plt.figure(figsize=(10, 6))
for ch in range(signal.shape[0]):
    plt.plot(freqs, power[ch], label=f'EEG{ch+1}', alpha=0.7)

plt.xlim(0, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title(f"Power Spectrum of Sample {sample_idx} (Label={label})")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.show()


X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


import torch.nn as nn
import torch.nn.functional as F

class SimpleEEGCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleEEGCNN, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.bn1= nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2= nn.BatchNorm1d(64)
        self.pool= nn.AdaptiveAvgPool1d(1)
        self.fc= nn.Linear(64, n_classes)

    def forward(self, x):  # x: [B, C, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # -> [B, 64]
        x = self.fc(x)
        return x

class TinyEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 16, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, n_classes)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyEEGNet(n_channels=X.shape[1], n_classes=len(np.unique(y))).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 训练 ===
for epoch in range(200):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

#test
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct/total:.3%}")
