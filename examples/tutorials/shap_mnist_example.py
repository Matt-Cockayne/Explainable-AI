"""
SHAP (SHapley Additive exPlanations) Tutorial
==============================================
This script demonstrates how to use SHAP to explain predictions of a CNN on MNIST dataset.

SHAP is based on Shapley values from game theory and provides a unified measure of feature importance:
1. Shapley values represent the average marginal contribution of a feature across all possible coalitions
2. SHAP satisfies desirable properties: local accuracy, missingness, and consistency
3. Different SHAP explainers exist: DeepExplainer, GradientExplainer, KernelExplainer
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import shap
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
        output = F.softmax(x, dim=1)  # Use softmax for SHAP
        return output


def train_model(model, device, train_loader, epochs=3):
    """Quick training function for demonstration"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')


def explain_with_shap_gradient(model, test_images, background_images, device):
    """Generate SHAP explanations using GradientExplainer"""
    model.eval()
    
    # GradientExplainer uses gradients to compute SHAP values
    explainer = shap.GradientExplainer(model, background_images)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(test_images)
    
    return shap_values


def explain_with_shap_deep(model, test_images, background_images, device):
    """Generate SHAP explanations using DeepExplainer (DeepLIFT approximation)"""
    model.eval()
    
    # DeepExplainer uses DeepLIFT algorithm
    explainer = shap.DeepExplainer(model, background_images)
    
    # Compute SHAP values
    # Note: check_additivity=False because dropout layers can cause small violations
    shap_values = explainer.shap_values(test_images, check_additivity=False)
    
    return shap_values


def visualize_shap_explanation(images, shap_values, predicted_classes, true_classes, method_name):
    """Visualize SHAP explanations for multiple images"""
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_images):
        image = images[idx].squeeze().cpu().numpy()
        pred_class = predicted_classes[idx]
        true_class = true_classes[idx]
        
        # Original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original\nTrue: {true_class}, Pred: {pred_class}')
        axes[idx, 0].axis('off')
        
        # SHAP values structure: list of arrays, one per sample
        # Each array has shape (1, H, W, n_classes) or similar
        # We need SHAP values for the predicted class
        shap_array = np.array(shap_values[idx])  # Get SHAP for this sample
        if len(shap_array.shape) == 4:  # (1, H, W, n_classes)
            shap_img = shap_array[0, :, :, pred_class]  # Extract predicted class
        elif len(shap_array.shape) == 3:  # (H, W, n_classes)
            shap_img = shap_array[:, :, pred_class]
        else:
            shap_img = shap_array.squeeze()
        
        # Positive SHAP values (red = positive contribution)
        axes[idx, 1].imshow(image, cmap='gray', alpha=0.5)
        im1 = axes[idx, 1].imshow(shap_img, cmap='Reds', alpha=0.6)
        axes[idx, 1].set_title(f'{method_name}\nPositive Attribution')
        axes[idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
        
        # Both positive and negative SHAP values
        axes[idx, 2].imshow(image, cmap='gray', alpha=0.3)
        im2 = axes[idx, 2].imshow(shap_img, cmap='seismic', alpha=0.7, 
                                  vmin=-np.abs(shap_img).max(), 
                                  vmax=np.abs(shap_img).max())
        axes[idx, 2].set_title(f'{method_name}\nAll Attribution\n(Red: +, Blue: -)')
        axes[idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
    
    plt.tight_layout()
    return fig


def compare_shap_methods(images, shap_values_gradient, shap_values_deep, 
                         predicted_classes, true_classes):
    """Compare GradientExplainer and DeepExplainer side by side"""
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_images):
        image = images[idx].squeeze().cpu().numpy()
        pred_class = predicted_classes[idx]
        true_class = true_classes[idx]
        
        # Original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original\nTrue: {true_class}, Pred: {pred_class}')
        axes[idx, 0].axis('off')
        
        # GradientExplainer
        shap_grad_array = np.array(shap_values_gradient[idx])
        if len(shap_grad_array.shape) == 4:  # (1, H, W, n_classes)
            shap_grad = shap_grad_array[0, :, :, pred_class]
        elif len(shap_grad_array.shape) == 3:  # (H, W, n_classes)
            shap_grad = shap_grad_array[:, :, pred_class]
        else:
            shap_grad = shap_grad_array.squeeze()
            
        axes[idx, 1].imshow(image, cmap='gray', alpha=0.3)
        im1 = axes[idx, 1].imshow(shap_grad, cmap='seismic', alpha=0.7,
                                  vmin=-np.abs(shap_grad).max(),
                                  vmax=np.abs(shap_grad).max())
        axes[idx, 1].set_title('GradientExplainer')
        axes[idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
        
        # DeepExplainer
        shap_deep_array = np.array(shap_values_deep[idx])
        if len(shap_deep_array.shape) == 4:  # (1, H, W, n_classes)
            shap_deep = shap_deep_array[0, :, :, pred_class]
        elif len(shap_deep_array.shape) == 3:  # (H, W, n_classes)
            shap_deep = shap_deep_array[:, :, pred_class]
        else:
            shap_deep = shap_deep_array.squeeze()
            
        axes[idx, 2].imshow(image, cmap='gray', alpha=0.3)
        im2 = axes[idx, 2].imshow(shap_deep, cmap='seismic', alpha=0.7,
                                  vmin=-np.abs(shap_deep).max(),
                                  vmax=np.abs(shap_deep).max())
        axes[idx, 2].set_title('DeepExplainer')
        axes[idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
    
    plt.tight_layout()
    return fig


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('./results/shap')
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
            if total >= 1000:
                break
    
    accuracy = 100. * correct / total
    print(f"\nModel Accuracy on test set: {accuracy:.2f}%")
    
    # Prepare background and test samples for SHAP
    print("\nPreparing data for SHAP...")
    background_samples = []
    test_samples = []
    test_labels = []
    
    train_iter = iter(train_loader)
    for _ in range(2):  # 128 background samples
        data, _ = next(train_iter)
        background_samples.append(data)
    background_data = torch.cat(background_samples, dim=0).to(device)
    
    test_iter = iter(test_loader)
    for _ in range(3):  # 3 test samples
        data, label = next(test_iter)
        test_samples.append(data)
        test_labels.append(label.item())
    test_data = torch.cat(test_samples, dim=0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(test_data)
        pred_labels = predictions.argmax(dim=1).cpu().numpy()
    
    print(f"Test samples - True labels: {test_labels}, Predicted: {pred_labels.tolist()}")
    
    # Method 1: GradientExplainer
    print("\nGenerating SHAP explanations using GradientExplainer...")
    shap_values_gradient = explain_with_shap_gradient(model, test_data, background_data, device)
    
    # Debug: print structure
    print(f"SHAP values structure: list of {len(shap_values_gradient)} arrays (one per class)")
    print(f"Each array shape: {shap_values_gradient[0].shape}")
    
    fig1 = visualize_shap_explanation(test_samples, shap_values_gradient, 
                                      pred_labels, test_labels, "GradientExplainer")
    output_path1 = output_dir / 'shap_gradient_explanations.png'
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path1}")
    plt.close()
    
    # Method 2: DeepExplainer
    print("\nGenerating SHAP explanations using DeepExplainer...")
    shap_values_deep = explain_with_shap_deep(model, test_data, background_data, device)
    
    fig2 = visualize_shap_explanation(test_samples, shap_values_deep,
                                      pred_labels, test_labels, "DeepExplainer")
    output_path2 = output_dir / 'shap_deep_explanations.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()
    
    # Comparison
    print("\nCreating comparison visualization...")
    fig3 = compare_shap_methods(test_samples, shap_values_gradient, shap_values_deep,
                                pred_labels, test_labels)
    output_path3 = output_dir / 'shap_methods_comparison.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path3}")
    plt.close()
    
    print("\n" + "="*70)
    print("SHAP Tutorial Complete!")
    print("="*70)
    print("\nKey Concepts:")
    print("1. SHAP is based on Shapley values from cooperative game theory")
    print("2. It fairly distributes the prediction among features")
    print("3. GradientExplainer: Uses gradients (faster, but approximate)")
    print("4. DeepExplainer: Uses DeepLIFT (more accurate for deep networks)")
    print("5. Red = positive contribution, Blue = negative contribution")
    print("6. SHAP values are additive: sum of all values = prediction difference")
    print("\nAdvantages over LIME:")
    print("- Theoretically grounded (game theory)")
    print("- Consistent and locally accurate")
    print("- Faster for deep learning (GradientExplainer)")


if __name__ == "__main__":
    main()
