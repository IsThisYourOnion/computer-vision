import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



def run_test_examples(model, test_loader, device=torch.device("cpu"), num_examples=5):
    """
    Run a few test examples through the trained model and display their classifications.

    Parameters:
    - model: The trained CNN model.
    - test_loader: DataLoader for the test dataset.
    - device: The device (CPU, GPU, or MPS) to run the evaluation on.
    - num_examples: Number of test examples to classify.
    """
    model.eval()  
    images, labels = next(iter(test_loader))  
    
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():  
        outputs = model(images[:num_examples])
        _, predicted = torch.max(outputs, 1)
    
    # Display the images and their predicted labels
    plt.figure(figsize=(10, 2))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        image_to_display = images[i].cpu().numpy().transpose((1, 2, 0))
        plt.imshow(image_to_display)
        plt.title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}")
        plt.axis('off')
    plt.show()


def load_model(model_class, load_path='models/artifacts/model.pth'):
    # Initialize the model structure (make sure it's the same structure as the saved model)
    model = model_class()
    # Load the state_dict into the model
    model.load_state_dict(torch.load(load_path))
    model.eval()  # Set the model to evaluation mode
    return model


