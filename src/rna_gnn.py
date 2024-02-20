import sys
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Callable, List, Optional
import torch
from torch_geometric.data import download_url
from model import GCN
import numpy as np 


dataset = torch.load("../data/york/processedyork.pt")

train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]

model = GCN(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train():
	model.train()
	for data in train_loader: 
		out = model(data.x, data.edge_index, data.batch)  
		loss = criterion(out, data.y)  
		loss.backward()  
		optimizer.step() 
		optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch) 
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')