import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import TensorDataset, DataLoader
print("Script started")
df = pd.read_csv("ENB2012_data.csv")
df.head()
features = df.iloc[:, 0:8]  # First 8 columns are input features
target = df["Y1"]  # Heating Load
#split the model before converting to tensors
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test= train_test_split(X_scaled, target, test_size=0.2, random_state=42)
#convert them to tensors to run the torch model
X_train_tensor= torch.tensor(X_train, dtype=torch.float32)
y_train_tensor= torch.tensor(y_train.values, dtype= torch.float32).view(-1, 1)
X_test_tensor= torch.tensor(X_test, dtype=torch.float32)
y_test_tensor= torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
#define torch model
class EnergyModel(nn.Module):
  def __init__(self, input_dim):
    super(EnergyModel, self).__init__()
    self.layers= nn.Sequential(
        nn.Linear(input_dim,64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

  def forward(self, x):
    return self.layers(x)

model= EnergyModel(input_dim=8)

#optimize the model using Adam
criterion = nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.01)
# Create PyTorch DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss {total_loss:.2f}")

# Visualization
plt.figure(figsize=(6,4))
plt.plot(range(1, len(losses)+1), losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.tight_layout()
plt.show()

#  Prediction block — place this LAST
new_sample = torch.tensor([[0.75, 750, 300, 200, 3.5, 2, 0.1, 3]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    predicted_load = model(new_sample).item()
    print(f"\nPredicted Heating Load for new sample: {predicted_load:.2f}")