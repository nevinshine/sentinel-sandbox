import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Data Processor (The "Feeder") ---
def load_and_process_data():
    print("📂 READING DATA: UNSW-NB15 CSVs...")
    # Load the CSVs you just downloaded
    train_df = pd.read_csv('data/unsw/UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('data/unsw/UNSW_NB15_testing-set.csv')

    # Drop ID and non-numeric garbage
    train_df = train_df.drop(['id', 'attack_cat'], axis=1) # Keep 'label' (0=Normal, 1=Attack)
    test_df = test_df.drop(['id', 'attack_cat'], axis=1)

    # Encode Text Columns (like "TCP", "UDP", "FIN") to numbers
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            le = LabelEncoder()
            # Fit on both to avoid "Unknown Category" errors
            combined_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
            le.fit(combined_data)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

    # Separate Labels (Targets) from Features (Input)
    y_train = train_df.pop('label').values
    x_train = train_df.values
    y_test = test_df.pop('label').values
    x_test = test_df.values

    # Normalize data to 0-1 range (Critical for Weightless Networks)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to PyTorch Tensors
    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.LongTensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.LongTensor(y_test)

    return tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test

# --- 2. The Weightless Model (CPU Optimized) ---
class WeightlessLayer(nn.Module):
    def __init__(self, num_inputs, address_size=4):
        super().__init__()
        self.address_size = address_size
        self.num_luts = num_inputs // address_size
        self.entry_per_lut = 2 ** address_size
        self.luts = nn.Embedding(self.num_luts * self.entry_per_lut, 1)
        nn.init.uniform_(self.luts.weight, -0.1, 0.1)
        self.register_buffer('powers', 2 ** torch.arange(address_size))
        self.register_buffer('offsets', torch.arange(self.num_luts) * self.entry_per_lut)

    def forward(self, x):
        batch_size = x.shape[0]
        limit = self.num_luts * self.address_size
        x = x[:, :limit]
        x_reshaped = x.view(batch_size, self.num_luts, self.address_size)
        addresses = (x_reshaped * self.powers).sum(dim=-1).long()
        final_indices = addresses + self.offsets
        return self.luts(final_indices).sum(dim=1)

class SecurityDWN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # Input features -> Binary representation
        self.layer1 = WeightlessLayer(input_features, address_size=4) 
        self.batch_norm = nn.BatchNorm1d(1)
        self.classifier = nn.Linear(1, 2) # Output: 2 classes (Normal vs Attack)

    def forward(self, x):
        # Binarize inputs (Thermometer encoding simulation)
        # If value > 0.5, bit is 1. Else 0.
        x = (x > 0.5).float() 
        out = self.layer1(x)
        out = self.batch_norm(out)
        out = self.classifier(out)
        return out

# --- 3. Training Loop ---
def train():
    x_train, y_train, x_test, y_test = load_and_process_data()
    print(f"✅ DATA LOADED. Features: {x_train.shape[1]} | Training Samples: {len(x_train)}")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SecurityDWN(input_features=x_train.shape[1]).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("🚀 SYSTEM: Training Security Model on UNSW-NB15...")

    for epoch in range(1, 6): # 5 Epochs
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Test Accuracy
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(test_dataset)
        print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f} | Accuracy={acc:.2f}%')

if __name__ == '__main__':
    train()
