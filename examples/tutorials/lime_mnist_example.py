"""
LIME (Local Interpretable Model-agnostic Explanations) Tutorial
================================================================
This script demonstrates how to use LIME to explain predictions of a CNN on MNIST dataset.

LIME approximates the model locally with an interpretable model (linear model) by:
1. Generating perturbed samples around the instance to explain
2. Getting predictions for these perturbed samples
3. Weighting samples by their proximity to the original instance
4. Fitting an interpretable model on this weighted dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pathlib import Path


# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model(model, device, train_loader, epochs=3):
    """Quick training function for demonstration"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')


def batch_predict(images, model, device):
    """Prediction function for LIME"""
    model.eval()
    batch = torch.stack([torch.from_numpy(img).float() for img in images])
    
    # LIME might pass images with channel dimension already, or as RGB
    # Handle different input shapes
    if len(batch.shape) == 3:  # (batch, H, W)
        batch = batch.unsqueeze(1)  # Add channel dimension -> (batch, 1, H, W)
    elif len(batch.shape) == 4 and batch.shape[-1] == 3:  # (batch, H, W, 3) RGB
        # Convert RGB to grayscale by averaging channels
        batch = batch.mean(dim=-1, keepdim=True)  # -> (batch, H, W, 1)
        batch = batch.permute(0, 3, 1, 2)  # -> (batch, 1, H, W)
    
    batch = batch.to(device)
    
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    
    return probs.cpu().numpy()


def explain_with_lime(model, image, device, num_samples=1000, num_features=5):
    """Generate LIME explanation for a single image"""
    explainer = lime_image.LimeImageExplainer()
    
    # LIME expects images in HWC format
    image_np = image.squeeze().cpu().numpy()
    
    # Get explanation
    explanation = explainer.explain_instance(
        image_np,
        lambda x: batch_predict(x, model, device),
        top_labels=3,
        hide_color=0,
        num_samples=num_samples
    )
    
    return explanation


def visualize_lime_explanation(image, explanation, predicted_class, true_class):
    """Visualize LIME explanation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title(f'Original Image\nTrue: {true_class}, Pred: {predicted_class}')
    axes[0].axis('off')
    
    # Get explanation for predicted class
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Positive features (supporting prediction)
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title('LIME: Positive Features\n(Support prediction)')
    axes[1].axis('off')
    
    # Show feature importance weights
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    axes[2].imshow(mark_boundaries(temp, mask))
    axes[2].set_title('LIME: All Important Features\n(Pos + Neg)')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('./results/lime')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Initialize and train model
    print("\nInitializing model...")
    model = SimpleCNN().to(device)
    
    print("\nTraining model (this may take a few minutes)...")
    train_model(model, device, train_loader, epochs=3)
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            if total >= 1000:  # Quick accuracy check
                break
    
    accuracy = 100. * correct / total
    print(f"\nModel Accuracy on test set: {accuracy:.2f}%")
    
    # Generate LIME explanations for a few examples
    print("\nGenerating LIME explanations...")
    num_examples = 3
    
    test_iter = iter(test_loader)
    for idx in range(num_examples):
        # Get a test image
        image, true_label = next(test_iter)
        image = image.to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(image)
            pred_label = output.argmax(dim=1).item()
        
        print(f"\nExample {idx+1}: True label: {true_label.item()}, Predicted: {pred_label}")
        
        # Generate LIME explanation
        print("  Generating LIME explanation (this takes ~30 seconds)...")
        explanation = explain_with_lime(model, image, device, num_samples=1000)
        
        # Visualize
        fig = visualize_lime_explanation(image, explanation, pred_label, true_label.item())
        output_path = output_dir / f'lime_explanation_example_{idx+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    print("\n" + "="*70)
    print("LIME Tutorial Complete!")
    print("="*70)
    print("\nKey Concepts:")
    print("1. LIME creates interpretable explanations by approximating the model locally")
    print("2. It generates perturbed samples and learns which features matter most")
    print("3. Green regions show features that support the prediction")
    print("4. LIME is model-agnostic - works with any classifier")
    print("5. Trade-off: slower (needs many forward passes) but very intuitive")


if __name__ == "__main__":
    main()
