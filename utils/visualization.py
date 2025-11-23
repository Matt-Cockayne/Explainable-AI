"""
Visualization utilities for comparing and displaying explanations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Union
import cv2


def apply_colormap(
    heatmap: torch.Tensor,
    colormap: str = 'jet',
    normalize: bool = True
) -> np.ndarray:
    """
    Apply colormap to heatmap.
    
    Args:
        heatmap: Heatmap tensor (H, W) or (1, H, W)
        colormap: Matplotlib colormap name
        normalize: Whether to normalize heatmap
        
    Returns:
        RGB image (H, W, 3) in range [0, 1]
    """
    heatmap_np = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    
    # Squeeze out batch dimension if present
    if heatmap_np.ndim == 3 and heatmap_np.shape[0] == 1:
        heatmap_np = heatmap_np.squeeze(0)
    
    if normalize:
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_np)[..., :3]  # Remove alpha channel, handle any shape
    
    return colored


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Input image (C, H, W) or (H, W, C), normalized to [0, 1]
        heatmap: Heatmap (H, W) or (1, H, W)
        alpha: Blending factor for heatmap
        colormap: Colormap to use
        
    Returns:
        Overlaid image (H, W, 3)
    """
    # Convert heatmap to numpy and squeeze any extra dimensions
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.squeeze(heatmap)  # Remove all singleton dimensions
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        # Squeeze batch dimension if present
        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np.squeeze(0)
        if image_np.shape[0] in [1, 3]:  # (C, H, W)
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.shape[2] == 1:  # Grayscale
            image_np = np.repeat(image_np, 3, axis=2)
    else:
        image_np = image
    
    # Denormalize if needed (check for ImageNet normalization)
    if image_np.min() < 0 or (image_np.max() < 1.0 and image_np.max() - image_np.min() < 0.5):
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    elif image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Apply colormap to heatmap
    heatmap_colored = apply_colormap(heatmap, colormap=colormap)
    
    # Resize if necessary
    if image_np.shape[:2] != heatmap_colored.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored,
            (image_np.shape[1], image_np.shape[0])
        )
    
    # Blend
    overlaid = (1 - alpha) * image_np + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def compare_methods(
    image: torch.Tensor,
    explanations: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    titles: Optional[List[str]] = None,
    colormap: str = 'jet',
    figsize: tuple = (20, 4)
) -> plt.Figure:
    """
    Create side-by-side comparison of multiple explanation methods.
    
    Args:
        image: Original image
        explanations: Dictionary mapping method names to heatmaps
        save_path: Path to save figure
        titles: Custom titles (defaults to method names)
        colormap: Colormap to use
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_methods = len(explanations)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)
    
    # Prepare image
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        if img_np.ndim == 4:
            img_np = img_np[0]  # Remove batch dimension
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
    else:
        img_np = image
    
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Show original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show explanations
    for idx, (method_name, heatmap) in enumerate(explanations.items()):
        overlaid = overlay_heatmap(image, heatmap, alpha=0.5, colormap=colormap)
        axes[idx + 1].imshow(overlaid)
        
        title = titles[idx] if titles and idx < len(titles) else method_name
        axes[idx + 1].set_title(title, fontsize=12, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_comparison(
    image: Union[torch.Tensor, np.ndarray],
    explanations: Dict[str, torch.Tensor],
    metrics_results: Optional[Dict[str, Dict]] = None,
    save_path: Optional[str] = None,
    colormap: str = 'jet',
    original_image: Optional[Union['Image.Image', np.ndarray]] = None
) -> plt.Figure:
    """
    Comprehensive visualization with explanations and metrics.
    
    Args:
        image: Original image (processed/upsampled)
        explanations: Dictionary of explanations
        metrics_results: Optional metrics for each method
        save_path: Path to save figure
        colormap: Colormap to use
        original_image: Optional original low-resolution image to display
        
    Returns:
        Matplotlib figure
    """
    n_methods = len(explanations)
    
    # Determine grid size
    if metrics_results:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, n_methods + 1, height_ratios=[3, 1])
    else:
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(20, 4))
        gs = None
    
    # Prepare image
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
    else:
        img_np = image
    
    # Denormalize if image appears to be normalized (negative values or small range)
    if img_np.min() < 0 or (img_np.max() < 1.0 and img_np.max() - img_np.min() < 0.5):
        # Apply ImageNet denormalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    elif img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Show original image
    if gs:
        ax_img = fig.add_subplot(gs[0, 0])
    else:
        ax_img = axes[0]
    
    # Use original low-res image if provided, otherwise use processed image
    if original_image is not None:
        from PIL import Image
        if isinstance(original_image, Image.Image):
            display_img = np.array(original_image)
        else:
            display_img = original_image
        # Convert grayscale to RGB if needed
        if display_img.ndim == 2 or (display_img.ndim == 3 and display_img.shape[2] == 1):
            if display_img.ndim == 3:
                display_img = display_img.squeeze()
            display_img = np.stack([display_img] * 3, axis=-1)
        if display_img.max() > 1.0:
            display_img = display_img / 255.0
        ax_img.imshow(display_img, interpolation='nearest')
        ax_img.set_title('Original Image (Low-Res)', fontsize=12, fontweight='bold')
    else:
        ax_img.imshow(img_np)
        ax_img.set_title('Original Image', fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Show explanations
    for idx, (method_name, heatmap) in enumerate(explanations.items()):
        if gs:
            ax = fig.add_subplot(gs[0, idx + 1])
        else:
            ax = axes[idx + 1]
        
        overlaid = overlay_heatmap(image, heatmap, alpha=0.5, colormap=colormap)
        ax.imshow(overlaid)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add metrics if available
        if metrics_results and method_name in metrics_results:
            ax_metrics = fig.add_subplot(gs[1, idx + 1])
            ax_metrics.axis('off')
            
            metrics = metrics_results[method_name]
            metrics_text = '\n'.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            ax_metrics.text(
                0.5, 0.5, metrics_text,
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_deletion_insertion_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot deletion and insertion curves for multiple methods.
    
    Args:
        results: Dictionary mapping method names to results with curves
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for method_name, result in results.items():
        if 'deletion_curve' in result:
            steps = np.arange(len(result['deletion_curve']))
            ax1.plot(
                steps, result['deletion_curve'],
                label=f"{method_name} (AUC: {result.get('deletion_auc', 0):.3f})",
                linewidth=2
            )
        
        if 'insertion_curve' in result:
            steps = np.arange(len(result['insertion_curve']))
            ax2.plot(
                steps, result['insertion_curve'],
                label=f"{method_name} (AUC: {result.get('insertion_auc', 0):.3f})",
                linewidth=2
            )
    
    ax1.set_xlabel('Number of Pixels Removed', fontsize=12)
    ax1.set_ylabel('Model Confidence', fontsize=12)
    ax1.set_title('Deletion Curve (Lower AUC = Better)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Number of Pixels Added', fontsize=12)
    ax2.set_ylabel('Model Confidence', fontsize=12)
    ax2.set_title('Insertion Curve (Higher AUC = Better)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    metrics_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create bar plots comparing metrics across methods.
    
    Args:
        metrics_results: Dictionary mapping method names to metrics
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Collect all unique metrics
    all_metrics = set()
    for metrics in metrics_results.values():
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    n_metrics = len(all_metrics)
    
    # Create subplots
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    methods = list(metrics_results.keys())
    
    for idx, metric_name in enumerate(all_metrics):
        ax = axes[idx]
        
        values = [
            metrics_results[method].get(metric_name, 0)
            for method in methods
        ]
        
        ax.bar(methods, values, alpha=0.7)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
