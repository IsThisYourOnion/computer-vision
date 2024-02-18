import torch
from ..models.cnn import CNN
from ..utils.data_helpers import loaders  
import torch.nn as nn
from torch import optim


def train_model(model, loaders, num_epochs=10, lr=0.001, save_path='models/artifacts/model.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in loaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(loaders["train"])}')
    
    print('Finished Training')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Initialize the model
model = CNN()
# Train the model
train_model(model, loaders)
