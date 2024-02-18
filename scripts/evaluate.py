import torch 
from train import model
from ..utils.data_helpers import loaders  
from ..utils.training_utils import load_model, run_test_examples 
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, test_loader, device=torch.device("cpu")):
    """
    Evaluate the model's performance on the test dataset.

    Parameters:
    - model: The trained CNN model.
    - test_loader: DataLoader for the test dataset.
    - device: The device (CPU, GPU, or MPS) to run the evaluation on.

    Returns:
    - The accuracy of the model on the test dataset.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize the count of correct predictions and the total number of predictions
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            
            # Select the class with the highest probability as our prediction
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate the accuracy
    accuracy = 100 * correct / total
    
    print(f'Accuracy of the model on the test images: {accuracy}%')
    
    return accuracy

# Set the device to MPS if available, otherwise to CPU
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
# Example usage
model = load_model(CNN, 'models/artifacts/model.pth')
# Move the model to the specified device
model.to(device)
# Evaluate the model
evaluate_model(model, loaders['test'], device=device)
run_test_examples(model, loaders['test'], device=device, num_examples=6)

