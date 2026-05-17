#!/usr/bin/env python3

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import copy

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

l_encoder = joblib.load('label_encoder.pkl')

# numpy arrays to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val,dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t,y_val_t)

# data loaders - Instead of feeding all 560 training samples into the model at once, DataLoader breaks the data into batches of 32 samples
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # shuffle to ensure the model doesnt accidentally learn order of samples.
val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False)

INPUT_SIZE  = X_train.shape[1]   # 12781
LATENT_DIM  = 16
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS  = 300

def add_noise(x, noise_factor=0.1):
    return x + torch.randn_like(x) * noise_factor

class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
 
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
 
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
 
            nn.Linear(64, latent_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
 
            nn.Linear(64, 512),
            nn.LayerNorm(512),
            nn.ELU(),
 
            nn.Linear(512, 2048),
            nn.LayerNorm(2048),
            nn.ELU(),
 
            nn.Linear(2048, output_size),
        )
    
    def forward(self, z):
        return self.net(z)

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, z):
        return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(latent_dim, input_size)
        self.classifier = ClassifierHead(latent_dim, num_classes)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        logits = self.classifier(z)
        return reconstructed, logits

    def encode(self, x):
        return self.encoder(x)

class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0
        self.best_state = None
 
    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience
 
    def restore(self, model):
        model.load_state_dict(self.best_state)

def evaluate(model, loader, mse_loss_fn, ce_loss_fn, lam, device, noisy=False):
    model.eval()
    total_mse, total_ce, correct = 0.0, 0.0, 0
    n_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_in = add_noise(X_batch) if noisy else X_batch
            recon, logits = model(X_in)
            total_mse += mse_loss_fn(recon, X_batch).item()
            total_ce  += ce_loss_fn(logits, y_batch).item()
            correct   += (logits.argmax(1) == y_batch).sum().item()
            n_batches += 1
    n = len(loader.dataset)
    return total_mse / n_batches, total_ce / n_batches, correct / n

def plot_curves(history, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_mse'], label='Train MSE')
    axes[0].plot(history['val_mse'],   label='Val MSE')
    axes[0].set_title(f'{title} — Reconstruction loss')
    axes[0].set_xlabel('Epoch'); axes[0].legend()
 
    axes[1].plot(history['train_ce'], label='Train CE')
    axes[1].plot(history['val_ce'],   label='Val CE')
    axes[1].set_title(f'{title} — Classification loss')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
 
    axes[2].plot(history['train_acc'], label='Train Acc')
    axes[2].plot(history['val_acc'],   label='Val Acc')
    axes[2].set_title(f'{title} — Accuracy')
    axes[2].set_xlabel('Epoch'); axes[2].legend()
 
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


model = Autoencoder(INPUT_SIZE, LATENT_DIM, NUM_CLASSES).to(DEVICE)
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
stopper = EarlyStopping(patience=15)

history_pretrain = {k: [] for k in ['train_mse', 'val_mse', 'train_ce', 'val_ce', 'train_acc', 'val_acc']}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_mse, train_ce, train_correct = 0.0, 0.0, 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        X_noisy = add_noise(X_batch)
 
        recon, logits = model(X_noisy)
        loss = mse_loss(recon, X_batch)   # reconstruction only
 
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
 
        train_mse     += loss.item()
        train_ce      += ce_loss(logits, y_batch).item()
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        n_batches     += 1

    val_mse, val_ce, val_acc = evaluate(model, val_loader, mse_loss, ce_loss, lam=0, device=DEVICE)
    t_mse = train_mse / n_batches
    t_ce  = train_ce  / n_batches
    t_acc = train_correct / len(train_loader.dataset)
 
    history_pretrain['train_mse'].append(t_mse)
    history_pretrain['val_mse'].append(val_mse)
    history_pretrain['train_ce'].append(t_ce)
    history_pretrain['val_ce'].append(val_ce)
    history_pretrain['train_acc'].append(t_acc)
    history_pretrain['val_acc'].append(val_acc)
 
    scheduler.step(val_mse)
 
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train MSE: {t_mse:.4f} | Val MSE: {val_mse:.4f} | Val Acc: {val_acc:.4f}")
 
    if stopper.step(val_mse, model):
        print(f"\nEarly stop at epoch {epoch+1}")
        stopper.restore(model)
        break
 
plot_curves(history_pretrain, 'Pretrain', 'pretrain_curves.png')
torch.save(model.state_dict(), 'ae_pretrained.pth')

print("\n── Regime 2: frozen encoder ─────────────────────────────────")

model_frozen = Autoencoder(INPUT_SIZE, LATENT_DIM, NUM_CLASSES).to(DEVICE)
model_frozen.load_state_dict(torch.load('ae_pretrained.pth'))

for param in model_frozen.encoder.parameters():
    param.requires_grad = False
for param in model_frozen.decoder.parameters():
    param.requires_grad = False

optimiser_frozen = optim.Adam(
    filter(lambda p: p.requires_grad, model_frozen.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler_frozen = optim.lr_scheduler.ReduceLROnPlateau(optimiser_frozen, patience=5, factor=0.5)
stopper_frozen   = EarlyStopping(patience=15)

history_frozen = {k: [] for k in ['train_mse', 'val_mse', 'train_ce', 'val_ce', 'train_acc', 'val_acc']}

for epoch in range(NUM_EPOCHS):
    model_frozen.train()
    train_mse, train_ce, train_correct = 0.0, 0.0, 0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        recon, logits = model_frozen(X_batch)
        loss = ce_loss(logits, y_batch)

        optimiser_frozen.zero_grad()
        loss.backward()
        optimiser_frozen.step()

        train_mse     += mse_loss(recon, X_batch).item()
        train_ce      += loss.item()
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        n_batches     += 1

    val_mse, val_ce, val_acc = evaluate(model_frozen, val_loader, mse_loss, ce_loss, lam=0, device=DEVICE, noisy=True)
    t_mse = train_mse / n_batches
    t_ce  = train_ce  / n_batches
    t_acc = train_correct / len(train_loader.dataset)

    history_frozen['train_mse'].append(t_mse)
    history_frozen['val_mse'].append(val_mse)
    history_frozen['train_ce'].append(t_ce)
    history_frozen['val_ce'].append(val_ce)
    history_frozen['train_acc'].append(t_acc)
    history_frozen['val_acc'].append(val_acc)

    scheduler_frozen.step(val_ce)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train CE: {t_ce:.4f} | Val CE: {val_ce:.4f} | Val Acc: {val_acc:.4f}")

    if stopper_frozen.step(val_ce, model_frozen):
        print(f"\nEarly stop at epoch {epoch+1}")
        stopper_frozen.restore(model_frozen)
        break

plot_curves(history_frozen, 'Frozen encoder', 'frozen_curves.png')

# confusion matrix + report
model_frozen.eval()
with torch.no_grad():
    _, logits = model_frozen(X_test_t.to(DEVICE))
    preds = logits.argmax(1).cpu().numpy()

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=l_encoder.classes_,
            yticklabels=l_encoder.classes_, cmap='BuPu')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Frozen encoder — confusion matrix')
plt.tight_layout()
plt.savefig('frozen_confusion.png', dpi=150)
plt.show()

print(f"\nMacro F1: {f1_score(y_test, preds, average='macro'):.4f}")
print(classification_report(y_test, preds, target_names=l_encoder.classes_))
torch.save(model_frozen.state_dict(), 'ae_frozen.pth')