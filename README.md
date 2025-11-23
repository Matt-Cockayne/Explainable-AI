# MedXAI: Medical Image Explainability Toolkit

A comprehensive PyTorch-based framework for explainable AI in medical imaging. MedXAI provides unified implementations of multiple explainability methods with quantitative evaluation metrics, specifically designed for medical image classification tasks.

## Overview

MedXAI bridges the gap between deep learning model predictions and clinical interpretability by offering:

- **Multiple XAI Methods**: Gradient-based, perturbation-based, and model-agnostic techniques
- **Quantitative Evaluation**: Faithfulness and localization metrics for objective assessment
- **Medical Dataset Support**: Pre-configured for MedMNIST datasets (DermaMNIST, PneumoniaMNIST, ChestMNIST)
- **Interactive Interface**: Gradio-based web application for real-time explanation generation
- **Tutorial Notebooks**: Step-by-step guides for each explainability method

## Key Features

### Explainability Methods

**Gradient-Based**
- **GradCAM**: Gradient-weighted Class Activation Mapping - fast, class-specific visualizations
- **GradCAM++**: Enhanced localization for multiple object instances
- **Integrated Gradients**: Attribution through path integration from baseline

**Perturbation-Based**
- **RISE**: Randomized Input Sampling for Explanation - model-agnostic importance estimation
- **LIME**: Local Interpretable Model-agnostic Explanations - superpixel-based local approximations
- **SHAP**: SHapley Additive exPlanations - game-theoretic feature attribution

### Evaluation Metrics

- **Deletion AUC**: Measures confidence degradation when removing important pixels (lower is better)
- **Insertion AUC**: Measures confidence improvement when adding important pixels (higher is better)
- **Faithfulness Metrics**: Quantifies correlation between attributions and model behavior
- **Pointing Game**: Evaluates localization accuracy against ground truth annotations

### Visualization & Interface

- **Comparative Visualizations**: Side-by-side heatmap overlays for method comparison
- **Quantitative Dashboards**: Performance curves and metric summaries
- **Web Interface**: Interactive Gradio application with real-time explanation generation
- **Export Functionality**: Save results as high-resolution images and structured data

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Matt-Cockayne/MedXAI.git
cd MedXAI

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `numpy>=1.24.0`, `scipy>=1.10.0`
- `matplotlib>=3.7.0`, `opencv-python>=4.8.0`
- `scikit-learn>=1.3.0`, `scikit-image>=0.21.0`
- `lime>=0.2.0`, `shap>=0.42.0`
- `medmnist>=3.0.0` (for medical datasets)
- `gradio>=4.0.0` (for web interface)

## Quick Start

### Basic Usage

```python
import torch
from utils import load_model, get_medical_dataset
from explainers import GradCAM, SHAP, LIME
from utils.visualization import visualize_comparison

# Load pre-trained model and dataset
model = load_model('resnet50', num_classes=7, device='cuda')
test_dataset = get_medical_dataset('dermamnist', split='test')

# Get a sample image
image, label = test_dataset[0]
image_batch = image.unsqueeze(0).to('cuda')

# Initialize explainers
explainers = {
    'GradCAM': GradCAM(model, 'layer4', device='cuda'),
    'LIME': LIME(model, device='cuda'),
    'SHAP': SHAP(model, device='cuda')
}

# Generate explanations
explanations = {}
for name, explainer in explainers.items():
    explanations[name] = explainer.explain(image_batch, target_class=label)

# Visualize results
fig = visualize_comparison(image, explanations)
```

### Launch Web Interface

```bash
# Start the interactive Gradio application
python interface/app.py

# Access at http://localhost:7860
```

### Tutorial Notebooks

Explore detailed tutorials in the `notebooks/` directory:

1. **GradCAM_Tutorial.ipynb** - Gradient-based visualization fundamentals
2. **LIME_Tutorial.ipynb** - Perturbation-based local explanations
3. **SHAP_Tutorial.ipynb** - Game-theoretic feature attribution
4. **DermaMNIST_Explainability_Tutorial.ipynb** - Skin lesion classification
5. **PneumoniaMNIST_Explainability_Tutorial.ipynb** - Pneumonia detection
6. **ChestMNIST_Explainability_Tutorial.ipynb** - Multi-disease classification

## Project Structure

```
MedXAI/
├── interface/
│   └── app.py                    # Gradio web application
├── explainers/
│   ├── __init__.py
│   ├── gradcam.py               # GradCAM implementation
│   ├── gradcam_plusplus.py      # GradCAM++ implementation
│   ├── integrated_gradients.py  # Integrated Gradients
│   ├── rise.py                  # RISE implementation
│   ├── lime_explainer.py        # LIME wrapper
│   └── shap_explainer.py        # SHAP wrapper
├── metrics/
│   ├── __init__.py
│   ├── deletion_insertion.py    # Deletion/Insertion AUC
│   └── faithfulness.py          # Faithfulness metrics
├── utils/
│   ├── __init__.py
│   ├── visualization.py         # Plotting and comparison tools
│   ├── data_utils.py            # Dataset loaders
│   └── model_utils.py           # Model loading utilities
├── notebooks/
│   ├── GradCAM_Tutorial.ipynb
│   ├── LIME_Tutorial.ipynb
│   ├── SHAP_Tutorial.ipynb
│   ├── DermaMNIST_Explainability_Tutorial.ipynb
│   ├── PneumoniaMNIST_Explainability_Tutorial.ipynb
│   └── ChestMNIST_Explainability_Tutorial.ipynb
├── examples/
│   └── tutorials/               # Standalone tutorial scripts
├── requirements.txt
└── README.md
```

## Advanced Usage

### Method Comparison

```python
from explainers import GradCAM, GradCAMPlusPlus, IntegratedGradients, RISE
from metrics import DeletionInsertion
from utils.visualization import plot_deletion_insertion_curves

# Initialize multiple explainers
explainers = {
    'GradCAM': GradCAM(model, 'layer4', device),
    'GradCAM++': GradCAMPlusPlus(model, 'layer4', device),
    'Integrated Gradients': IntegratedGradients(model, device),
    'RISE': RISE(model, device, n_masks=1000)
}

# Generate explanations
explanations = {}
for name, explainer in explainers.items():
    explanations[name] = explainer.explain(image_batch, target_class=label)

# Quantitative evaluation
di_metric = DeletionInsertion(model, device, n_steps=50)
results = {}
for name, heatmap in explanations.items():
    results[name] = di_metric.evaluate(image_batch, heatmap, label)

# Plot comparison curves
plot_deletion_insertion_curves(results, save_path='metrics_comparison.png')
```

### Custom Dataset Integration

```python
from torch.utils.data import Dataset
from torchvision import transforms

class CustomMedicalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob('*.png'))
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)
    
    def __len__(self):
        return len(self.image_paths)

# Use with MedXAI
dataset = CustomMedicalDataset('path/to/images')
```

### Batch Processing

```python
from pathlib import Path
from tqdm import tqdm

# Process multiple images
output_dir = Path('results/batch_explanations')
output_dir.mkdir(parents=True, exist_ok=True)

for idx, (image, label) in enumerate(tqdm(test_dataset)):
    image_batch = image.unsqueeze(0).to(device)
    
    # Generate explanations
    explanations = {}
    for name, explainer in explainers.items():
        explanations[name] = explainer.explain(image_batch, target_class=label)
    
    # Save visualization
    fig = visualize_comparison(image, explanations)
    fig.savefig(output_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
```

## Supported Architectures & Datasets

### Models
- **ResNet Family**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **Vision Transformers**: ViT-B/16, ViT-L/16 (attention-based explanations)
- **EfficientNet**: EfficientNet-B0 through B7
- **DenseNet**: DenseNet121, DenseNet169, DenseNet201
- **Custom Models**: Any PyTorch CNN with accessible convolutional layers

### Medical Imaging Datasets

**MedMNIST Suite** (via `medmnist` package):
- **DermaMNIST**: 7-class skin lesion classification (28×28 RGB)
- **PneumoniaMNIST**: Binary pneumonia detection (28×28 grayscale)
- **ChestMNIST**: 14-class thoracic disease classification (28×28 grayscale)
- **OrganMNIST**: 11-class organ segmentation
- **PathMNIST**: 9-class histopathology classification

All datasets are automatically downloaded and preprocessed with standardized transforms.

### Performance Characteristics

| Method | Speed | Resolution | Model Access | Best Use Case |
|--------|-------|------------|--------------|---------------|
| GradCAM | Very Fast (~10ms) | Feature map size | Gradients required | Quick CNN explanations |
| GradCAM++ | Fast (~15ms) | Feature map size | Gradients required | Multiple object localization |
| Integrated Gradients | Fast (~50ms) | Input size | Gradients required | Attribution completeness |
| RISE | Slow (~30s) | Input size | Black-box | Model-agnostic |
| LIME | Very Slow (~60s) | Superpixel | Black-box | Local interpretability |
| SHAP | Fast (~100ms) | Input size | Gradients preferred | Consistent attributions |

## Evaluation Metrics

### Deletion AUC
Measures the degradation in model confidence as the most important pixels (according to the explanation) are progressively removed. Lower values indicate better explanations, as important pixels should strongly influence predictions.

### Insertion AUC  
Measures the improvement in model confidence as the most important pixels are progressively added to a blank baseline. Higher values indicate better explanations, demonstrating that identified regions are sufficient for accurate classification.

### Faithfulness Metrics
Quantifies how well explanations reflect the model's actual decision-making process through correlation between attribution values and output changes under perturbations.

### Usage Example

```python
from metrics import DeletionInsertion, FaithfulnessMetrics

# Initialize metrics
di_metric = DeletionInsertion(model, device, n_steps=50)
faith_metric = FaithfulnessMetrics(model, device)

# Evaluate explanations
di_results = di_metric.evaluate(image_batch, heatmap, target_class)
print(f"Deletion AUC: {di_results['deletion_auc']:.3f}")
print(f"Insertion AUC: {di_results['insertion_auc']:.3f}")

faith_results = faith_metric.evaluate_all(image_batch, heatmap, target_class)
for metric_name, value in faith_results.items():
    print(f"{metric_name}: {value:.3f}")
```

## Clinical Interpretation Guidelines

When applying XAI methods to medical imaging:

1. **Validate with Domain Experts**: Always have clinicians review explanations for clinical relevance
2. **Compare Multiple Methods**: Use at least 2-3 different XAI techniques to identify consistent patterns
3. **Consider Context**: Explanations should align with known disease manifestations and anatomical structures
4. **Assess Quantitatively**: Use deletion/insertion metrics to validate explanation quality objectively
5. **Document Limitations**: Clearly communicate when model focus deviates from clinical expectations

### Common Pitfalls
- **Spurious Correlations**: Models may focus on artifacts or metadata rather than pathology
- **Resolution Limitations**: Low-resolution heatmaps may miss fine-grained features
- **Class Confusion**: Ensure target class is specified correctly for multi-class problems
- **Batch Effects**: Dataset-specific biases may influence what models learn to focus on

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{MedXAI,
  author = {Matthew Cockayne},
  title = {MedXAI: Medical Image Explainability Toolkit},
  year = {2025},
  url = {https://github.com/Matt-Cockayne/MedXAI}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Areas for improvement:
- Additional explainability methods (e.g., Attention Rollout, LayerCAM)
- More evaluation metrics (e.g., pointing game, sensitivity-n)
- Support for 3D medical imaging
- Integration with clinical DICOM workflows

Please submit issues or pull requests on GitHub.

## Acknowledgments

This toolkit builds upon foundational work in explainable AI and medical imaging:
- GradCAM paper by Selvaraju et al. (2017)
- LIME by Ribeiro et al. (2016)
- SHAP by Lundberg & Lee (2017)
- MedMNIST dataset collection by Yang et al. (2021)

Developed as part of PhD research in trustworthy AI for healthcare applications.
