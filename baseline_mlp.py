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

# print(f"X_train:{X_train.shape}") #(560, 12781)
# print(f"X_val:{X_val.shape}") #(120, 12781)
# print(f"X_test:{X_test.shape}") #(121, 12781)


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

input_size = X_train.shape[1]  
num_classes = 5 

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1 compress from 12781 -> 512
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), #dropout reduced bc only 64 neurons 
            
            # Output layer 64 -> 5 classes
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

model = MLP(input_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()  # standard loss function for multi class classifincation 
optimiser = optim.Adam(model.parameters(), lr=1e-3) # most widely used optimiser for nn's 

num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

#The data reaches 100% validation accuracy really fast...so adding early stopping, called patience-based early stopping. 
patience = 10 # no improvement for 10 consecutive epochs 
min_delta = 1e-4 # stop early if validation loss failed to improve by at least min_delta = 1e-4 for 10 consecutive epochs.
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training phase
    model.train()  # activates dropout 
    train_loss, train_correct = 0, 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        #forward pass 
        predictions = model(X_batch)
        loss = loss_function(predictions, y_batch)
        
        #backward pass 
        optimiser.zero_grad()  #clear gradients from last step
        loss.backward() #compute gradients
        optimiser.step() #update weights
        
        train_loss += loss.item()
        train_correct += (predictions.argmax(1) == y_batch).sum().item()
    
    # Validation 
    model.eval()  # disables dropout 
    val_loss, val_correct = 0, 0
    
    with torch.no_grad():  
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            
            val_loss += loss.item()
            val_correct += (predictions.argmax(1) == y_batch).sum().item()
    
    # Record metrics 
    avg_train_loss = train_loss/len(train_loader)
    avg_val_loss = val_loss/len(val_loader)
    train_acc = train_correct/len(X_train)
    val_acc = val_correct/len(X_val)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    if (epoch+1)%10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    #early stopping stuff 
    if avg_val_loss<best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict()) 
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1} "
              f"(no val loss improvement for {patience} epochs)")
        model.load_state_dict(best_model_state) 
        break
        
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('MLP Training and Validation Loss')
ax1.legend()

# Accuracy curves
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('MLP Training and Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('mlp_training_curves.png', dpi=150)
plt.show()

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    X_test_tensor = X_test_t.to(device)
    outputs = model(X_test_tensor)
    preds = outputs.argmax(1).cpu().numpy()

all_preds = preds
all_labels = y_test

# Macro F1 and full report
print("\n── MLP Test Results ──────────────────────────────")
print(f"Macro F1 Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
print("\nFull Classification Report:")
print(classification_report(all_labels, all_preds, 
                             target_names=l_encoder.classes_))

# confusion matrix
c_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(c_matrix, annot=True, fmt='d', 
            xticklabels=l_encoder.classes_,
            yticklabels=l_encoder.classes_,
            cmap='BuPu')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('MLP Confusion Matrix')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=150)
plt.show()

torch.save(model.state_dict(), 'mlp_baseline.pth')
