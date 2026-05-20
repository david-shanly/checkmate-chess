import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ChessCNN

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("/content/drive/MyDrive/chess-engine/dataset.npz")

X = data["X"]
y = data["y"]

print("Dataset loaded:", X.shape)

# -----------------------------
# Train / Validation Split
# -----------------------------
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

# -----------------------------
# Convert to tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# DataLoader
# -----------------------------
BATCH_SIZE = 256

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = ChessCNN().to(device)

# Loss functions
value_loss_fn = nn.MSELoss()
policy_loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# Training Settings
# -----------------------------
EPOCHS = 10

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        value_pred, policy_pred = model(X_batch)

        # Value loss
        value_loss = value_loss_fn(value_pred, y_batch)

        # Dummy policy targets (temporary)
        policy_target = torch.zeros(policy_pred.shape[0],
                                    dtype=torch.long).to(device)

        policy_loss = policy_loss_fn(policy_pred, policy_target)

        loss = value_loss + policy_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            value_pred, policy_pred = model(X_batch)

            value_loss = value_loss_fn(value_pred, y_batch)

            policy_target = torch.zeros(policy_pred.shape[0],
                                        dtype=torch.long).to(device)

            policy_loss = policy_loss_fn(policy_pred, policy_target)

            loss = value_loss + policy_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f}"
    )

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "model.pt")

print("\nTraining complete.")
print("Model saved as model.pt")