"""
GradCAM (Gradient-weighted Class Activation Mapping) Tutorial
==============================================================
This script demonstrates how to use GradCAM to explain predictions of a CNN on MNIST dataset.

GradCAM generates visual explanations by:
1. Computing gradients of the target class with respect to feature maps
2. Global average pooling these gradients to get importance weights
3. Weighted combination of feature maps followed by ReLU
4. Upsampling to original image size

Key advantage: Works with any CNN architecture without modification or retraining.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2
from pathlib import Path


# Simple CNN for MNIST with named layers for GradCAM
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Store feature maps and gradients for GradCAM
        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Save feature maps for GradCAM
        self.feature_maps = x
        
        # Register hook to capture gradients
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def save_gradient(self, grad):
        """Hook to save gradients"""
        self.gradients = grad


class GradCAM:
    """GradCAM implementation"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_cam(self, image, target_class=None):
        """
        Generate GradCAM heatmap for the target class
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class
        
        Returns:
            cam: GradCAM heatmap (H, W)
            predicted_class: Predicted class index
        """
        # Forward pass
        image = image.to(self.device)
        image.requires_grad = True
        
        output = self.model(image)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and feature maps
        gradients = self.model.gradients.cpu().data.numpy()[0]  # (C, H, W)
        feature_maps = self.model.feature_maps.cpu().data.numpy()[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of feature maps
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        
        # Apply ReLU (only positive influences)
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class
    
    def visualize_cam(self, image, cam, predicted_class, true_class):
        """
        Visualize GradCAM heatmap overlaid on original image
        
        Args:
            image: Original image tensor
            cam: GradCAM heatmap
            predicted_class: Predicted class
            true_class: True class
        """
        # Convert image to numpy
        img_np = image.squeeze().cpu().numpy()
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Normalize image for visualization
        img_normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255
        
        # Overlay heatmap on image
        img_rgb = np.stack([img_normalized] * 3, axis=-1)
        overlay = 0.5 * img_rgb + 0.5 * heatmap
        overlay = overlay / overlay.max()
        
        return img_normalized, cam_resized, heatmap, overlay


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


def visualize_gradcam_examples(model, test_loader, device, num_examples=5):
    """Generate and visualize GradCAM for multiple examples"""
    gradcam = GradCAM(model, device)
    
    fig, axes = plt.subplots(num_examples, 5, figsize=(20, 4*num_examples))
    
    test_iter = iter(test_loader)
    
    for idx in range(num_examples):
        # Get test image
        image, true_label = next(test_iter)
        
        # Generate GradCAM
        cam, pred_label = gradcam.generate_cam(image, target_class=None)
        
        # Visualize
        img_normalized, cam_resized, heatmap, overlay = gradcam.visualize_cam(
            image, cam, pred_label, true_label.item()
        )
        
        # Plot
        axes[idx, 0].imshow(img_normalized, cmap='gray')
        axes[idx, 0].set_title(f'Original\nTrue: {true_label.item()}\nPred: {pred_label}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam_resized, cmap='jet')
        axes[idx, 1].set_title('GradCAM Heatmap')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(heatmap)
        axes[idx, 2].set_title('Heatmap (Jet)')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('GradCAM Overlay')
        axes[idx, 3].axis('off')
        
        # Show only high-confidence regions
        high_conf_mask = cam_resized > 0.5
        masked_img = img_normalized.copy()
        masked_img[~high_conf_mask] = 0
        axes[idx, 4].imshow(masked_img, cmap='gray')
        axes[idx, 4].set_title('High Confidence\nRegions (>0.5)')
        axes[idx, 4].axis('off')
    
    plt.tight_layout()
    return fig


def compare_correct_vs_incorrect(model, test_loader, device):
    """Compare GradCAM for correct vs incorrect predictions"""
    gradcam = GradCAM(model, device)
    
    correct_examples = []
    incorrect_examples = []
    
    # Find examples
    test_iter = iter(test_loader)
    while len(correct_examples) < 3 or len(incorrect_examples) < 3:
        image, true_label = next(test_iter)
        
        # Get prediction
        with torch.no_grad():
            output = model(image.to(device))
            pred_label = output.argmax(dim=1).item()
        
        if pred_label == true_label.item() and len(correct_examples) < 3:
            correct_examples.append((image, true_label.item(), pred_label))
        elif pred_label != true_label.item() and len(incorrect_examples) < 3:
            incorrect_examples.append((image, true_label.item(), pred_label))
    
    # Visualize
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    # Correct predictions
    for idx, (image, true_label, pred_label) in enumerate(correct_examples):
        cam, _ = gradcam.generate_cam(image, target_class=pred_label)
        img_normalized, cam_resized, heatmap, overlay = gradcam.visualize_cam(
            image, cam, pred_label, true_label
        )
        
        axes[0, idx*2].imshow(img_normalized, cmap='gray')
        axes[0, idx*2].set_title(f'Correct: {pred_label}')
        axes[0, idx*2].axis('off')
        
        axes[0, idx*2+1].imshow(overlay)
        axes[0, idx*2+1].set_title('GradCAM')
        axes[0, idx*2+1].axis('off')
    
    # Incorrect predictions
    for idx, (image, true_label, pred_label) in enumerate(incorrect_examples):
        cam, _ = gradcam.generate_cam(image, target_class=pred_label)
        img_normalized, cam_resized, heatmap, overlay = gradcam.visualize_cam(
            image, cam, pred_label, true_label
        )
        
        axes[1, idx*2].imshow(img_normalized, cmap='gray')
        axes[1, idx*2].set_title(f'Wrong: True={true_label}\nPred={pred_label}')
        axes[1, idx*2].axis('off')
        
        axes[1, idx*2+1].imshow(overlay)
        axes[1, idx*2+1].set_title('GradCAM')
        axes[1, idx*2+1].axis('off')
    
    fig.suptitle('GradCAM: Correct vs Incorrect Predictions', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('./results/gradcam')
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
    
    # Generate GradCAM visualizations
    print("\nGenerating GradCAM explanations...")
    fig1 = visualize_gradcam_examples(model, test_loader, device, num_examples=5)
    output_path1 = output_dir / 'gradcam_examples.png'
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path1}")
    plt.close()
    
    # Compare correct vs incorrect predictions
    print("\nComparing correct vs incorrect predictions...")
    fig2 = compare_correct_vs_incorrect(model, test_loader, device)
    output_path2 = output_dir / 'gradcam_correct_vs_incorrect.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()
    
    print("\n" + "="*70)
    print("GradCAM Tutorial Complete!")
    print("="*70)
    print("\nKey Concepts:")
    print("1. GradCAM uses gradients flowing into the last convolutional layer")
    print("2. Global average pooling of gradients gives feature importance weights")
    print("3. Weighted combination of feature maps creates the localization map")
    print("4. ReLU ensures only positive influences (neurons that increase score)")
    print("5. Heatmap shows which regions the model 'looks at' for prediction")
    print("\nAdvantages:")
    print("- Fast (single backward pass)")
    print("- Works with any CNN architecture")
    print("- No model modification or retraining needed")
    print("- Class-discriminative (specific to target class)")
    print("\nInterpretation:")
    print("- Bright regions = high importance for the predicted class")
    print("- Dark regions = low importance")
    print("- For incorrect predictions, GradCAM shows where model focused wrongly")


if __name__ == "__main__":
    main()
