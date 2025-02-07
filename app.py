# Converted from Jupyter Notebook to Python script

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define dataset class
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts):
        self.source_texts = source_texts
        self.target_texts = target_texts
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        return self.source_texts[idx], self.target_texts[idx]

# Load and preprocess dataset
data_path = "dataset.csv"  # Change this to the actual dataset file

data_df = pd.read_csv(data_path)
source_texts = data_df['source'].tolist()
target_texts = data_df['target'].tolist()

# Split dataset into train and test
train_src, test_src, train_tgt, test_tgt = train_test_split(
    source_texts, target_texts, test_size=0.2, random_state=42
)

# Define DataLoader
train_dataset = TranslationDataset(train_src, train_tgt)
test_dataset = TranslationDataset(test_src, test_tgt)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define translation model (Corrected to match original implementation)
class SimpleTranslationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(SimpleTranslationModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.relu(self.linear1(embedded))
        output = self.linear2(hidden)
        return output

# Define model parameters
input_dim = 5000  # Example vocabulary size
output_dim = 5000  # Example vocabulary size

model = SimpleTranslationModel(input_dim, output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Save model
torch.save(model.state_dict(), "translation_model.pth")
