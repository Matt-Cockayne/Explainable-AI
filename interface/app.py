"""
Interactive Gradio interface for MedMNIST explainability demonstrations.

Three pre-configured pages for:
1. DermaMNIST - Skin lesion classification
2. PneumoniaMNIST - Pneumonia detection  
3. ChestMNIST - Thoracic disease classification
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainers import (
    GradCAM, GradCAMPlusPlus, IntegratedGradients, RISE
)
from metrics import DeletionInsertion, FaithfulnessMetrics
from utils import (
    load_model, get_medical_dataset, visualize_comparison,
    overlay_heatmap
)

# Import for MNIST tutorial examples
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import cv2


class MedMNISTExplainabilityApp:
    """Interactive application for MedMNIST explainability demonstrations."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datasets = {}
        self.models = {}
        self.explainers = {}
        self.data_dir = Path('./data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # MNIST tutorial components
        self.mnist_model = None
        self.mnist_train_loader = None
        self.mnist_test_loader = None
        
    def load_dataset(self, dataset_name: str, num_classes: int, is_grayscale: bool = False):
        """Load MedMNIST dataset and initialize model."""
        try:
            # Import medmnist
            import medmnist
            
            # Custom transform for datasets
            if is_grayscale:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                transform = None  # Use default
            
            # Load dataset
            dataset = get_medical_dataset(
                dataset_name,
                root=str(self.data_dir),
                split='test',
                transform=transform,
                download=True
            )
            
            self.datasets[dataset_name] = dataset
            
            # Load model
            model = load_model('resnet50', num_classes=num_classes, device=self.device)
            self.models[dataset_name] = model
            
            # Initialize explainers
            self.explainers[dataset_name] = {
                'GradCAM': GradCAM(model, 'layer4', self.device),
                'GradCAM++': GradCAMPlusPlus(model, 'layer4', self.device),
                'Integrated Gradients': IntegratedGradients(model, self.device),
                'RISE': RISE(model, self.device, n_masks=1000),
                'LIME': 'lime_placeholder',  # Special handling in generate_explanations
                'SHAP': 'shap_placeholder'   # Special handling in generate_explanations
            }
            
            return f"Loaded {dataset_name}: {len(dataset)} test images, {num_classes} classes"
        except Exception as e:
            return f"Error loading {dataset_name}: {str(e)}"
    
    def load_sample(self, dataset_name: str, sample_idx: int, class_names: List[str]) -> Tuple:
        """Load a sample from dataset."""
        try:
            if dataset_name not in self.datasets:
                return None, "Dataset not loaded", None, None
            
            dataset = self.datasets[dataset_name]
            model = self.models[dataset_name]
            
            # Get sample
            image, label = dataset[sample_idx]
            
            # Handle multi-label (ChestMNIST)
            if isinstance(label, np.ndarray) and len(label.shape) > 0 and len(label) > 1:
                positive_labels = np.where(label == 1)[0]
                if len(positive_labels) > 0:
                    label = int(positive_labels[0])
                else:
                    label = 0
            else:
                # Convert numpy array to scalar properly
                label = int(label.item()) if isinstance(label, np.ndarray) else int(label)
            
            # Get original low-res image
            import medmnist
            from medmnist import INFO
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            original_dataset = DataClass(split='test', download=False, root=str(self.data_dir), transform=None)
            original_image = original_dataset[sample_idx][0]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(image_batch)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_class = output.argmax(dim=1).item()
                confidence = probs[pred_class].item()
            
            true_label = class_names[label] if label < len(class_names) else f"Class {label}"
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
            
            info_text = f"Sample #{sample_idx}\nTrue: {true_label}\nPredicted: {pred_label} ({confidence:.2%})"
            
            return original_image, info_text, image_batch, label
            
        except Exception as e:
            return None, f"Error: {str(e)}", None, None
    
    def generate_explanations(
        self,
        dataset_name: str,
        image_batch: torch.Tensor,
        label: int,
        methods: List[str]
    ) -> List[Tuple[Image.Image, str]]:
        """Generate explanations and metrics."""
        try:
            if dataset_name not in self.explainers:
                return [(None, "Dataset not loaded")] * 4
            
            model = self.models[dataset_name]
            explainers = self.explainers[dataset_name]
            
            results = []
            di_metric = DeletionInsertion(model, self.device, n_steps=30)
            
            for method_name in methods[:4]:  # Max 4 methods
                try:
                    # Handle LIME specially
                    if method_name == 'LIME':
                        heatmap = self._generate_lime_for_medical(image_batch, model)
                    # Handle SHAP specially - returns complete visualization
                    elif method_name == 'SHAP':
                        shap_visual = self._generate_shap_for_medical(image_batch, model, label)
                        # SHAP returns RGB image directly, convert to PIL
                        img = Image.fromarray(shap_visual.astype(np.uint8))
                        
                        # For SHAP, we can't compute standard metrics since it's a visualization
                        metrics_text = f"{method_name}\n(Visual only)"
                        results.append((img, metrics_text))
                        continue  # Skip standard overlay and metrics
                    else:
                        explainer = explainers[method_name]
                        
                        # Generate explanation
                        if method_name == 'RISE':
                            heatmap = explainer.explain(image_batch, label, n_masks=500)
                        else:
                            heatmap = explainer.explain(image_batch, label)
                    
                    # Ensure heatmap is detached and on CPU
                    if isinstance(heatmap, torch.Tensor):
                        heatmap = heatmap.detach().cpu()
                    
                    # Create matplotlib visualization matching SHAP size
                    img_np = image_batch.squeeze().detach().cpu().numpy()
                    if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                        img_np = img_np.transpose(1, 2, 0)
                    
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    # Convert heatmap to numpy if needed
                    if isinstance(heatmap, torch.Tensor):
                        heatmap_np = heatmap.cpu().numpy()
                    else:
                        heatmap_np = np.array(heatmap)
                    
                    # Create figure matching SHAP size
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(img_np)
                    im = ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
                    ax.set_title(f'{method_name}', fontsize=12)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    plt.tight_layout()
                    
                    # Convert to image
                    from io import BytesIO
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    img = Image.open(buf)
                    plt.close()
                    
                    # Compute metrics
                    result = di_metric.evaluate(image_batch, heatmap, label)
                    metrics_text = f"{method_name}\nDel: {result['deletion_auc']:.3f}\nIns: {result['insertion_auc']:.3f}"
                    
                    results.append((img, metrics_text))
                except Exception as e:
                    results.append((None, f"{method_name}\nError: {str(e)}"))
            
            # Pad to 4 results
            while len(results) < 4:
                results.append((None, ""))
            
            return results
            
        except Exception as e:
            error_result = (None, f"Error: {str(e)}")
            return [error_result] * 4
    
    def _generate_lime_for_medical(self, image_batch: torch.Tensor, model: nn.Module) -> np.ndarray:
        """Generate LIME explanation for medical images."""
        # Convert to numpy for LIME
        img_np = image_batch.squeeze().detach().cpu().numpy()
        
        # Handle different image formats
        if len(img_np.shape) == 3:  # (C, H, W)
            img_np = img_np.transpose(1, 2, 0)  # (H, W, C)
        
        # Normalize to [0, 1] for LIME
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        def batch_predict(images):
            model.eval()
            # Convert back to tensors
            batch_list = []
            for img in images:
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)  # Convert to RGB
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
                batch_list.append(img_tensor)
            
            batch = torch.cat(batch_list, dim=0).to(self.device)
            
            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=500
        )
        
        # Get top predicted class
        pred_class = explanation.top_labels[0]
        
        # Extract explanation as heatmap
        _, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=10,
            hide_rest=False,
            min_weight=0.0
        )
        
        # Convert mask to heatmap (resize to match input)
        heatmap = mask.astype(np.float32)
        
        # Resize heatmap to match input image dimensions (224x224)
        target_size = (224, 224)
        if heatmap.shape != target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        
        return heatmap
    
    def _generate_shap_for_medical(self, image_batch: torch.Tensor, model: nn.Module, label: int) -> np.ndarray:
        """Generate SHAP explanation for medical images."""
        # Get background samples (use a small batch)
        background = torch.randn(10, *image_batch.shape[1:]).to(self.device) * 0.1
        
        # Create SHAP explainer
        explainer = shap.GradientExplainer(model, background)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(image_batch)
        
        # SHAP returns a list where each element corresponds to an OUTPUT CLASS
        # shap_values[class_idx] gives array of shape (batch, C_in, H, W)
        # Following MNIST example pattern:
        shap_array = np.array(shap_values[0])  # Get first element (list structure)
        
        # Extract SHAP values for the predicted class
        # shap_array has shape like (batch, C_in, H, W, n_classes) or (C_in, H, W, n_classes)
        if len(shap_array.shape) == 5:  # (batch, C_in, H, W, n_classes)
            shap_img = shap_array[0, :, :, :, label]  # Get batch 0, all channels, predicted class
            # Sum across input channels (C_in)
            shap_img = np.sum(shap_img, axis=0)  # (H, W)
        elif len(shap_array.shape) == 4:  # (C_in, H, W, n_classes) or (batch, H, W, n_classes)
            # Check if first dim is batch (1) or channels (3)
            if shap_array.shape[0] == 1:  # batch dimension
                shap_img = shap_array[0, :, :, label]  # (H, W)
            else:  # channel dimension
                shap_img = shap_array[:, :, :, label]  # (C_in, H, W)
                shap_img = np.sum(shap_img, axis=0)  # Sum channels -> (H, W)
        elif len(shap_array.shape) == 3:  # (H, W, n_classes)
            shap_img = shap_array[:, :, label]
        else:
            # Fallback - try to extract something meaningful
            shap_img = shap_array.squeeze()
            if len(shap_img.shape) > 2:
                shap_img = np.sum(shap_img, axis=0)
        
        # Ensure we have 2D array
        while len(shap_img.shape) > 2:
            shap_img = np.sum(shap_img, axis=0)
        
        # Resize if needed
        if shap_img.shape != (224, 224):
            if shap_img.size > 0 and min(shap_img.shape) > 0:
                shap_img = cv2.resize(shap_img, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                shap_img = np.zeros((224, 224), dtype=np.float32)
        
        # Get original image for overlay
        img_np = image_batch.squeeze().detach().cpu().numpy()
        
        # Convert from (C, H, W) to (H, W, C) for visualization
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        
        # Denormalize image (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Create visualization EXACTLY like MNIST tutorial
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Show original image with transparency (matching MNIST alpha=0.3)
        ax.imshow(img_np, alpha=0.3)
        
        # Overlay SHAP with seismic colormap (matching MNIST alpha=0.7)
        im = ax.imshow(shap_img, cmap='seismic', alpha=0.7,
                      vmin=-np.abs(shap_img).max(),
                      vmax=np.abs(shap_img).max())
        ax.set_title('SHAP Attribution\n(Red: +, Blue: -)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        # Convert to image
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf)
        plt.close()
        
        # Convert PIL image back to numpy array
        result_array = np.array(result_img)
        
        # Return the RGB array directly (complete visualization)
        return result_array
    
    # ==================== MNIST Tutorial Methods ====================
    
    def load_mnist_model(self):
        """Load and train MNIST model for tutorial demonstrations."""
        try:
            from torch.utils.data import DataLoader
            
            # Define SimpleCNN
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super(SimpleCNN, self).__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, 1)
                    self.conv2 = nn.Conv2d(32, 64, 3, 1)
                    self.dropout1 = nn.Dropout2d(0.25)
                    self.dropout2 = nn.Dropout2d(0.5)
                    self.fc1 = nn.Linear(9216, 128)
                    self.fc2 = nn.Linear(128, 10)
                    
                    # For GradCAM
                    self.feature_maps = None
                    self.gradients = None
                
                def forward(self, x):
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = self.conv2(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x, 2)
                    
                    # Save for GradCAM
                    self.feature_maps = x
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
                    self.gradients = grad
            
            # Load MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            from torchvision import datasets
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            
            self.mnist_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            self.mnist_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            
            # Train model
            self.mnist_model = SimpleCNN().to(self.device)
            optimizer = torch.optim.Adam(self.mnist_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            self.mnist_model.train()
            for epoch in range(3):
                for batch_idx, (data, target) in enumerate(self.mnist_train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.mnist_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 100 == 0:
                        yield f"Training... Epoch {epoch+1}/3, Batch {batch_idx}/{len(self.mnist_train_loader)}, Loss: {loss.item():.4f}"
            
            # Evaluate
            self.mnist_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.mnist_test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.mnist_model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    if total >= 1000:
                        break
            
            accuracy = 100. * correct / total
            yield f"✓ Model trained! Accuracy: {accuracy:.2f}%"
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def generate_lime_explanation(self, sample_idx: int):
        """Generate LIME explanation for MNIST sample."""
        try:
            if self.mnist_model is None:
                return None, None, "Please train the model first"
            
            # Get sample directly from test dataset
            from torchvision import datasets
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
            image, true_label = test_dataset[sample_idx]
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Prediction
            self.mnist_model.eval()
            with torch.no_grad():
                output = self.mnist_model(image.to(self.device))
                pred_label = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_label].item()
            
            # LIME explanation
            def batch_predict(images):
                self.mnist_model.eval()
                batch = torch.stack([torch.from_numpy(img).float() for img in images])
                
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(1)
                elif len(batch.shape) == 4 and batch.shape[-1] == 3:
                    batch = batch.mean(dim=-1, keepdim=True)
                    batch = batch.permute(0, 3, 1, 2)
                
                batch = batch.to(self.device)
                with torch.no_grad():
                    logits = self.mnist_model(batch)
                    probs = F.softmax(logits, dim=1)
                return probs.cpu().numpy()
            
            explainer = lime_image.LimeImageExplainer()
            image_np = image.squeeze().cpu().numpy()
            
            explanation = explainer.explain_instance(
                image_np,
                batch_predict,
                top_labels=3,
                hide_color=0,
                num_samples=1000
            )
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title(f'Original Image\nTrue: {true_label}, Pred: {pred_label} ({confidence:.2%})', fontsize=12)
            axes[0].axis('off')
            
            temp, mask = explanation.get_image_and_mask(pred_label, positive_only=True, num_features=5, hide_rest=False)
            axes[1].imshow(mark_boundaries(temp, mask))
            axes[1].set_title('LIME: Positive Features\n(Support Prediction)', fontsize=12)
            axes[1].axis('off')
            
            temp, mask = explanation.get_image_and_mask(pred_label, positive_only=False, num_features=10, hide_rest=False)
            axes[2].imshow(mark_boundaries(temp, mask))
            axes[2].set_title('LIME: All Features\n(Positive + Negative)', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save to PIL
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            result_img = Image.open(buf)
            plt.close()
            
            info = f"**LIME Explanation**\n\n"
            info += f"- True Label: {true_label}\n"
            info += f"- Predicted: {pred_label} ({confidence:.2%})\n"
            info += f"- Green boundaries show superpixels (regions) that support the prediction\n"
            info += f"- LIME perturbed 1000 samples to build local linear approximation"
            
            return image_np, result_img, info
            
        except Exception as e:
            return None, None, f"Error: {str(e)}"
    
    def generate_shap_explanation(self, sample_idx: int):
        """Generate SHAP explanation for MNIST sample."""
        try:
            if self.mnist_model is None:
                return None, None, "Please train the model first"
            
            # Get sample directly from test dataset
            from torchvision import datasets
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
            image, true_label = test_dataset[sample_idx]
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Prediction
            self.mnist_model.eval()
            with torch.no_grad():
                output = self.mnist_model(image.to(self.device))
                pred_label = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_label].item()
            
            # Background samples
            background_samples = []
            train_iter = iter(self.mnist_train_loader)
            for _ in range(2):
                data, _ = next(train_iter)
                background_samples.append(data)
            background_data = torch.cat(background_samples, dim=0).to(self.device)
            
            # SHAP GradientExplainer
            explainer = shap.GradientExplainer(self.mnist_model, background_data)
            shap_values = explainer.shap_values(image.to(self.device))
            
            # Extract SHAP values for predicted class
            shap_array = np.array(shap_values[0])
            if len(shap_array.shape) == 4:
                shap_img = shap_array[0, :, :, pred_label]
            elif len(shap_array.shape) == 3:
                shap_img = shap_array[:, :, pred_label]
            else:
                shap_img = shap_array.squeeze()
            
            image_np = image.squeeze().cpu().numpy()
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title(f'Original Image\nTrue: {true_label}, Pred: {pred_label} ({confidence:.2%})', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(image_np, cmap='gray', alpha=0.5)
            im1 = axes[1].imshow(shap_img, cmap='Reds', alpha=0.6)
            axes[1].set_title('SHAP: Positive Attribution\n(Increases Prediction)', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            
            axes[2].imshow(image_np, cmap='gray', alpha=0.3)
            im2 = axes[2].imshow(shap_img, cmap='seismic', alpha=0.7,
                                vmin=-np.abs(shap_img).max(),
                                vmax=np.abs(shap_img).max())
            axes[2].set_title('SHAP: All Attribution\n(Red: +, Blue: -)', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)
            
            plt.tight_layout()
            
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            result_img = Image.open(buf)
            plt.close()
            
            info = f"**SHAP Explanation**\n\n"
            info += f"- True Label: {true_label}\n"
            info += f"- Predicted: {pred_label} ({confidence:.2%})\n"
            info += f"- Based on Shapley values from game theory\n"
            info += f"- Red regions: positive contribution to prediction\n"
            info += f"- Blue regions: negative contribution\n"
            info += f"- SHAP values are additive and satisfy consistency"
            
            return image_np, result_img, info
            
        except Exception as e:
            return None, None, f"Error: {str(e)}"
    
    def generate_gradcam_explanation(self, sample_idx: int):
        """Generate GradCAM explanation for MNIST sample."""
        try:
            if self.mnist_model is None:
                return None, None, "Please train the model first"
            
            # Get sample directly from test dataset
            from torchvision import datasets
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
            image, true_label = test_dataset[sample_idx]
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            self.mnist_model.eval()
            image_input = image.to(self.device)
            image_input.requires_grad = True
            
            output = self.mnist_model(image_input)
            pred_label = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_label].item()
            
            # Backward pass
            self.mnist_model.zero_grad()
            target = output[0, pred_label]
            target.backward()
            
            # Get gradients and feature maps
            gradients = self.mnist_model.gradients.cpu().data.numpy()[0]
            feature_maps = self.mnist_model.feature_maps.cpu().data.numpy()[0]
            
            # GradCAM
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature_maps[i]
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            image_np = image.squeeze().cpu().numpy()
            
            # Resize and overlay
            cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
            img_normalized = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = heatmap.astype(np.float32) / 255
            
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
            overlay = 0.5 * img_rgb + 0.5 * heatmap
            overlay = overlay / overlay.max()
            
            # Visualize
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(img_normalized, cmap='gray')
            axes[0].set_title(f'Original Image\nTrue: {true_label}, Pred: {pred_label} ({confidence:.2%})', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('GradCAM Heatmap\n(Gradient-weighted)', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('GradCAM Overlay\n(Focused Regions)', fontsize=12)
            axes[2].axis('off')
            
            high_conf_mask = cam_resized > 0.5
            masked_img = img_normalized.copy()
            masked_img[~high_conf_mask] = 0
            axes[3].imshow(masked_img, cmap='gray')
            axes[3].set_title('High Confidence Regions\n(Threshold > 0.5)', fontsize=12)
            axes[3].axis('off')
            
            plt.tight_layout()
            
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            result_img = Image.open(buf)
            plt.close()
            
            info = f"**GradCAM Explanation**\n\n"
            info += f"- True Label: {true_label}\n"
            info += f"- Predicted: {pred_label} ({confidence:.2%})\n"
            info += f"- Highlights regions model focuses on for classification\n"
            info += f"- Bright areas = high importance\n"
            info += f"- Uses gradients from last convolutional layer\n"
            info += f"- Fast: single forward + backward pass"
            
            return image_np, result_img, info
            
        except Exception as e:
            return None, None, f"Error: {str(e)}"







def create_dataset_tab(app, dataset_name: str, num_classes: int, class_names: List[str], 
                       is_grayscale: bool = False, description: str = ""):
    """Create a tab for a specific MedMNIST dataset."""
    
    # State variables
    current_image_batch = gr.State(None)
    current_label = gr.State(None)
    
    with gr.Column():
        gr.Markdown(f"### {description}")
        
        # Load dataset button
        with gr.Row():
            load_btn = gr.Button(f"Load {dataset_name.upper()} Dataset", variant="primary", scale=2)
            status_text = gr.Textbox(label="Status", scale=3, interactive=False)
        
        load_btn.click(
            fn=lambda: app.load_dataset(dataset_name, num_classes, is_grayscale),
            outputs=[status_text]
        )
        
        # Sample selection
        with gr.Row():
            sample_slider = gr.Slider(
                minimum=0, maximum=1000, step=1, value=42,
                label="Select Sample Index"
            )
            load_sample_btn = gr.Button("Load Sample", variant="secondary")
        
        # Display sample and info
        with gr.Row():
            sample_image = gr.Image(label="Original Image (28x28)", type="pil", height=200)
            sample_info = gr.Textbox(label="Sample Information", lines=4)
        
        # Explainability methods
        gr.Markdown("### Generate Explanations")
        gr.Markdown("**LIME and SHAP are always shown. Select up to 2 additional methods:**")
        method_checkboxes = gr.CheckboxGroup(
            choices=['GradCAM', 'GradCAM++', 'Integrated Gradients', 'RISE'],
            value=['GradCAM', 'GradCAM++'],
            label="Additional Methods (select up to 2)"
        )
        
        explain_btn = gr.Button("Generate All 4 Explanations (LIME, SHAP + 2 selected)", variant="primary")
        
        # Results display
        with gr.Row():
            with gr.Column():
                result1_img = gr.Image(label="LIME Explanation", type="pil")
                result1_text = gr.Textbox(label="Metrics", lines=3)
            with gr.Column():
                result2_img = gr.Image(label="SHAP Explanation", type="pil")
                result2_text = gr.Textbox(label="Metrics", lines=3)
        
        with gr.Row():
            with gr.Column():
                result3_img = gr.Image(label="Method 3", type="pil")
                result3_text = gr.Textbox(label="Metrics", lines=3)
            with gr.Column():
                result4_img = gr.Image(label="Method 4", type="pil")
                result4_text = gr.Textbox(label="Metrics", lines=3)
        
        # Wire up callbacks
        def load_sample_wrapper(idx):
            img, info, batch, lbl = app.load_sample(dataset_name, int(idx), class_names)
            return img, info, batch, lbl
        
        load_sample_btn.click(
            fn=load_sample_wrapper,
            inputs=[sample_slider],
            outputs=[sample_image, sample_info, current_image_batch, current_label]
        )
        
        def explain_wrapper(batch, label, methods):
            if batch is None:
                empty = (None, "Load a sample first")
                return [empty] * 8
            # Always use LIME and SHAP first, then add selected methods (limit to 2)
            all_methods = ['LIME', 'SHAP'] + methods[:2]
            results = app.generate_explanations(dataset_name, batch, label, all_methods)
            # Unpack tuples for outputs
            outputs = []
            for img, text in results:
                outputs.extend([img, text])
            return outputs
        
        explain_btn.click(
            fn=explain_wrapper,
            inputs=[current_image_batch, current_label, method_checkboxes],
            outputs=[
                result1_img, result1_text, result2_img, result2_text,
                result3_img, result3_text, result4_img, result4_text
            ]
        )


def create_interface():
    """Create the Gradio interface with three MedMNIST dataset tabs."""
    app = MedMNISTExplainabilityApp()
    
    with gr.Blocks(title="MedMNIST Explainability Toolkit") as demo:
        gr.Markdown("""
        # 🔬 Explainable AI for Medical Imaging
        ### Interactive MedMNIST Explainability Demonstrations
        
        Explore medical imaging datasets with comprehensive explainability analysis.
        """)
        
        with gr.Tab("📚 Tutorial: LIME, SHAP & GradCAM"):
            gr.Markdown("""
            ## Educational Tutorial: Understanding XAI Methods
            
            This interactive tutorial demonstrates three fundamental explainability methods on MNIST handwritten digits.
            Each method offers unique insights into how neural networks make predictions.
            
            ### 🎯 Learning Objectives
            - Understand how LIME, SHAP, and GradCAM work
            - Compare different explanation approaches
            - Interpret visual explanations of model predictions
            
            ---
            """)
            
            # Training section
            with gr.Row():
                train_btn = gr.Button("🚀 Step 1: Train MNIST Model (3 epochs)", variant="primary", size="lg")
            
            training_output = gr.Textbox(label="Training Progress", lines=8, interactive=False)
            
            train_btn.click(
                fn=lambda: "\n".join(list(app.load_mnist_model())),
                outputs=[training_output]
            )
            
            gr.Markdown("---")
            
            # Method comparison theory
            with gr.Accordion("📖 Method Comparison & Theory", open=False):
                gr.Markdown("""
                ### Comparison of XAI Methods
                
                | Method | Type | Speed | Advantages | Best For |
                |--------|------|-------|------------|----------|
                | **LIME** | Model-agnostic | Slow (~30s) | Works with any model, intuitive | General purpose, any architecture |
                | **SHAP** | Gradient-based | Fast (~1s) | Theoretically grounded, consistent | Deep learning models |
                | **GradCAM** | CNN-specific | Very Fast (<0.1s) | Excellent visualizations | Convolutional networks |
                
                ### 🔍 LIME (Local Interpretable Model-agnostic Explanations)
                **How it works:**
                1. Perturbs the input image by turning superpixels on/off
                2. Gets model predictions for perturbed samples
                3. Weights samples by proximity to original
                4. Fits a linear model to approximate behavior locally
                
                **Key Insight:** Shows which image regions (superpixels) matter most for the prediction.
                
                ### 🎲 SHAP (SHapley Additive exPlanations)
                **How it works:**
                1. Based on Shapley values from cooperative game theory
                2. Computes fair attribution of prediction to each pixel
                3. Uses gradients to efficiently calculate contributions
                4. Satisfies desirable properties: local accuracy, missingness, consistency
                
                **Key Insight:** Each pixel gets a "credit" for its contribution to the prediction.
                
                ### 🎨 GradCAM (Gradient-weighted Class Activation Mapping)
                **How it works:**
                1. Computes gradients of target class w.r.t. last conv layer
                2. Global average pools gradients to get importance weights
                3. Weighted sum of feature maps creates localization map
                4. ReLU keeps only positive influences
                
                **Key Insight:** Highlights which regions the CNN focuses on for classification.
                """)
            
            gr.Markdown("---")
            gr.Markdown("## 🎮 Step 2: Generate Explanations")
            
            with gr.Row():
                sample_slider = gr.Slider(0, 100, value=42, step=1, label="Select MNIST Sample")
            
            with gr.Tab("LIME Analysis"):
                gr.Markdown("""
                ### 🟢 LIME: Local Interpretable Model-agnostic Explanations
                
                LIME explains predictions by approximating the model locally with an interpretable linear model.
                Green boundaries show superpixels (image regions) that support the prediction.
                
                **Interpretation Guide:**
                - **Left:** Original digit image with true/predicted labels
                - **Middle:** Positive features (regions that increase confidence in prediction)
                - **Right:** All important features (both positive and negative contributions)
                """)
                
                lime_btn = gr.Button("Generate LIME Explanation (~30 seconds)", variant="primary")
                
                with gr.Row():
                    lime_img = gr.Image(label="LIME Visualization", type="pil", height=400)
                
                lime_btn.click(
                    fn=lambda idx: app.generate_lime_explanation(int(idx))[1],
                    inputs=[sample_slider],
                    outputs=[lime_img]
                )
            
            with gr.Tab("SHAP Analysis"):
                gr.Markdown("""
                ### 🎲 SHAP: SHapley Additive exPlanations
                
                SHAP provides theoretically grounded explanations based on game theory.
                Colors indicate pixel contributions: **Red = positive**, **Blue = negative**.
                
                **Interpretation Guide:**
                - **Left:** Original digit image
                - **Middle:** Positive attributions (pixels that increase prediction score)
                - **Right:** All attributions (red increases score, blue decreases score)
                
                SHAP values are **additive**: the sum of all contributions equals the difference 
                between the prediction and the baseline (average prediction).
                """)
                
                shap_btn = gr.Button("Generate SHAP Explanation (~5 seconds)", variant="primary")
                
                with gr.Row():
                    shap_img = gr.Image(label="SHAP Visualization", type="pil", height=400)
                
                shap_btn.click(
                    fn=lambda idx: app.generate_shap_explanation(int(idx))[1],
                    inputs=[sample_slider],
                    outputs=[shap_img]
                )
            
            with gr.Tab("GradCAM Analysis"):
                gr.Markdown("""
                ### 🎨 GradCAM: Gradient-weighted Class Activation Mapping
                
                GradCAM visualizes which regions CNNs focus on when making predictions.
                Bright colors indicate high importance for the predicted class.
                
                **Interpretation Guide:**
                - **Panel 1:** Original digit image
                - **Panel 2:** GradCAM heatmap (bright = important regions)
                - **Panel 3:** Heatmap overlaid on original image
                - **Panel 4:** Only high-confidence regions (threshold > 0.5)
                
                GradCAM is **very fast** (single forward + backward pass) and shows 
                where the model "looks" when classifying the digit.
                """)
                
                gradcam_btn = gr.Button("Generate GradCAM Explanation (<1 second)", variant="primary")
                
                with gr.Row():
                    gradcam_img = gr.Image(label="GradCAM Visualization", type="pil", height=400)
                
                gradcam_btn.click(
                    fn=lambda idx: app.generate_gradcam_explanation(int(idx))[1],
                    inputs=[sample_slider],
                    outputs=[gradcam_img]
                )
            
            gr.Markdown("""
            ---
            ### 💡 Key Takeaways
            
            1. **LIME** is model-agnostic and intuitive but slow
            2. **SHAP** is theoretically grounded with guaranteed properties
            3. **GradCAM** is fast and provides excellent spatial visualizations for CNNs
            4. Different methods can highlight different aspects of the decision
            5. Use multiple methods for robust understanding of model behavior
            
            ### 📚 Further Reading
            - LIME: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
            - SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
            - GradCAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
            """)
        
        with gr.Tab("DermaMNIST - Skin Lesions"):
            create_dataset_tab(
                app,
                dataset_name='dermamnist',
                num_classes=7,
                class_names=[
                    'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
                    'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions'
                ],
                is_grayscale=False,
                description="**7-class skin lesion classification from dermatoscopic images**"
            )
        
        with gr.Tab("PneumoniaMNIST - Chest X-rays"):
            create_dataset_tab(
                app,
                dataset_name='pneumoniamnist',
                num_classes=2,
                class_names=['Normal', 'Pneumonia'],
                is_grayscale=True,
                description="**Binary pneumonia detection from pediatric chest X-rays**"
            )
        
        with gr.Tab("ChestMNIST - Thoracic Diseases"):
            create_dataset_tab(
                app,
                dataset_name='chestmnist',
                num_classes=14,
                class_names=[
                    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                    'Pleural Thickening', 'Hernia'
                ],
                is_grayscale=True,
                description="**14-class thoracic disease classification from NIH ChestX-ray14**"
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Tool
            
            Interactive demonstration of explainability methods on MedMNIST datasets.
            
            ### Datasets
            
            - **DermaMNIST**: 7 skin lesion types from HAM10000 dataset
            - **PneumoniaMNIST**: Binary pneumonia detection
            - **ChestMNIST**: 14 thoracic diseases from NIH ChestX-ray14
            
            ### Explainability Methods
            
            - **GradCAM**: Gradient-weighted Class Activation Mapping
            - **GradCAM++**: Improved pixel-wise weighting
            - **Integrated Gradients**: Path-based attribution
            - **RISE**: Randomized Input Sampling for Explanation
            - **LIME**: Local Interpretable Model-agnostic Explanations
            - **SHAP**: SHapley Additive exPlanations from game theory
            
            ### Metrics
            
            - **Deletion AUC**: Lower is better (explanation captures important features)
            - **Insertion AUC**: Higher is better (explanation is sufficient)
            
            ### Usage
            
            1. Click "Load Dataset" to download and initialize
            2. Select a sample using the slider
            3. Click "Load Sample" to view the image
            4. Choose explainability methods (up to 4)
            5. Click "Generate Explanations & Metrics"
            
            ### Citation
            
            ```bibtex
            @software{explainable_ai_toolkit,
              author = {Matthew Cockayne},
              title = {Explainable-AI: Medical Imaging Explainability Toolkit},
              year = {2025},
              url = {https://github.com/Matt-Cockayne/Explainable-AI}
            }
            ```
            
            ### References
            
            MedMNIST: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", Scientific Data, 2023
            """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Explainability Interface')
    parser.add_argument('--share', action='store_true', 
                        help='Create a public shareable link (for remote/headless environments)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run on (default: 7860)')
    parser.add_argument('--server', type=str, default="0.0.0.0",
                        help='Server address (default: 0.0.0.0)')
    args = parser.parse_args()
    
    demo = create_interface()
    
    # For headless/remote: use share=True to get a public URL
    # For local: use share=False
    demo.launch(
        share=args.share,
        server_name=args.server,
        server_port=args.port,
        show_error=True
    )
