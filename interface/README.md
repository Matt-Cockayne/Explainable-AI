# MedXAI Interactive Interface

## Overview

The Gradio-based web interface provides interactive demonstrations of explainability methods for medical imaging.

## Visual Examples

### GradCAM Explanations

**Multiple Examples Comparison**
![GradCAM Examples](../results/gradcam/gradcam_examples.png)
*GradCAM visualizations across multiple MNIST samples showing heatmaps, overlays, and high-confidence regions.*

**Correct vs Incorrect Predictions**
![GradCAM Correct vs Incorrect](../results/gradcam/gradcam_correct_vs_incorrect.png)
*Comparison of model attention for correct predictions (top row) vs incorrect predictions (bottom row).*

### LIME Explanations

**LIME Superpixel Analysis**
![LIME Example 1](../results/lime/lime_explanation_example_1.png)
![LIME Example 2](../results/lime/lime_explanation_example_2.png)
*LIME explanations showing original images, positive/negative features, and superpixel boundaries.*

### SHAP Explanations

**SHAP Methods Comparison**
![SHAP Comparison](../results/shap/shap_methods_comparison.png)
*Comparison of different SHAP explainer types: GradientExplainer vs DeepExplainer.*

**SHAP Gradient Explanations**
![SHAP Gradient](../results/shap/shap_gradient_explanations.png)
*SHAP gradient-based attributions showing positive (red) and negative (blue) feature contributions.*

### Medical Dataset Results

**DermaMNIST Skin Lesion Classification**
![DermaMNIST Comparison](../results/dermamnist/sample_42_comparison.png)
*Multi-method comparison for skin lesion classification: GradCAM, GradCAM++, Integrated Gradients, and RISE.*

![DermaMNIST Curves](../results/dermamnist/sample_42_curves.png)
*Deletion and Insertion curves showing faithfulness metrics for each explainability method.*

## Features

### 📚 Tutorial Tab: LIME, SHAP & GradCAM

**New educational page** that provides:

1. **Interactive Training**: Train a CNN on MNIST digits with live progress updates
2. **Three XAI Methods**: Side-by-side comparison of LIME, SHAP, and GradCAM
3. **Visual Explanations**: High-quality visualizations with detailed interpretations
4. **Theory & Context**: Collapsible accordion with method comparison table and theory

#### What Makes This Tutorial Page Great:

- **Clear Visual Design**: Each method has its own tab with 3-4 panel visualizations
- **Educational Content**: 
  - Comparison table showing speed, advantages, and use cases
  - Theory explanations for each method
  - Interpretation guides for understanding outputs
  - Key takeaways and further reading

- **Interactive Elements**:
  - Sample slider to explore different digits
  - One-click explanation generation
  - Live training progress
  - Estimated processing time for each method

#### Tutorial Structure:

1. **Step 1: Train Model**
   - Train SimpleCNN on MNIST (3 epochs)
   - Real-time training updates
   - Final accuracy display

2. **Step 2: Generate Explanations**
   - **LIME Tab**: Shows superpixel-based local explanations (~30s)
     - Original image with predictions
     - Positive features (green boundaries)
     - All important features
   
   - **SHAP Tab**: Game theory-based attributions (~5s)
     - Original image
     - Positive attributions (red heatmap)
     - All attributions (red/blue seismic colormap)
   
   - **GradCAM Tab**: CNN visualization (<1s)
     - Original image
     - GradCAM heatmap
     - Overlay visualization
     - High-confidence regions

3. **Theory Section** (Collapsible)
   - Comparison table
   - How each method works
   - Key insights for interpretation

### 🏥 Medical Dataset Tabs

Three pre-configured tabs for medical imaging:

1. **DermaMNIST**: Skin lesion classification (7 classes)
2. **PneumoniaMNIST**: Pneumonia detection (binary)
3. **ChestMNIST**: Thoracic diseases (14 classes)

Each tab provides:
- Dataset loading
- Sample browsing
- Multiple explainability methods (GradCAM, GradCAM++, Integrated Gradients, RISE)
- Faithfulness metrics (Deletion/Insertion AUC)

## Usage

### Launch the Interface

```bash
# Local usage
python app.py

# With public shareable link (for remote/headless environments)
python app.py --share

# Custom port
python app.py --port 8080

# Custom server address
python app.py --server 127.0.0.1
```

### Tutorial Workflow

1. Navigate to "📚 Tutorial: LIME, SHAP & GradCAM" tab
2. Click "🚀 Step 1: Train MNIST Model" and wait for training to complete (~2-3 minutes)
3. Use the sample slider to select a digit (0-100)
4. Click on each method's tab (LIME, SHAP, GradCAM)
5. Generate explanations for the selected sample
6. Compare different methods' outputs
7. Read the interpretation guides to understand the visualizations

### Medical Dataset Workflow

1. Select a dataset tab (DermaMNIST, PneumoniaMNIST, or ChestMNIST)
2. Click "Load Dataset" to download and initialize
3. Use the sample slider to select an image
4. Click "Load Sample" to view the image
5. Select explainability methods (up to 4)
6. Click "Generate Explanations & Metrics"
7. Review visualizations and metrics

## Technical Details

### Tutorial Implementation

- **Model**: SimpleCNN (2 conv layers, 2 FC layers, dropout)
- **Dataset**: MNIST (28x28 grayscale digits)
- **Training**: 3 epochs, Adam optimizer, CrossEntropyLoss
- **LIME**: 1000 perturbed samples, superpixel segmentation
- **SHAP**: GradientExplainer with 128 background samples
- **GradCAM**: Gradients from last conv layer (conv2)

### Visualization Features

- **High DPI**: All plots rendered at 150 DPI for clarity
- **Color Schemes**:
  - LIME: Green boundaries on grayscale
  - SHAP: Red (positive) / Blue (negative) on seismic colormap
  - GradCAM: Jet colormap with overlay
- **Informative Layouts**: Multi-panel comparisons with labeled axes
- **Real-time Generation**: Explanations generated on-demand

### Performance

| Method | Processing Time | Description |
|--------|----------------|-------------|
| LIME | ~30 seconds | Perturbs 1000 samples |
| SHAP | ~5 seconds | Gradient-based computation |
| GradCAM | <1 second | Single forward + backward pass |

## Dependencies

```
gradio>=4.0.0
torch>=2.0.0
torchvision
numpy
matplotlib
pillow
lime
shap
opencv-python
scikit-image
medmnist
```

## Interface Design Principles

1. **Progressive Disclosure**: Advanced theory in collapsible sections
2. **Clear Visual Hierarchy**: Emojis, headers, and spacing guide the user
3. **Educational Focus**: Every visualization includes interpretation guidance
4. **Interactive Learning**: Hands-on experimentation with instant feedback
5. **Comparison-Friendly**: Side-by-side method tabs for easy comparison

## File Structure

```
interface/
├── app.py              # Main Gradio interface
├── README.md           # This file
└── requirements.txt    # Dependencies (if separate from main repo)
```

## Future Enhancements

Potential additions:
- [ ] Side-by-side comparison of all three methods on same sample
- [ ] Quantitative metrics for tutorial (faithfulness, sensitivity)
- [ ] More advanced XAI methods (Grad-CAM++, Score-CAM)
- [ ] Export functionality for visualizations
- [ ] Batch processing mode
- [ ] Custom image upload for tutorial

## Citation

If you use this interface in your research or teaching:

```bibtex
@software{medxai_interface,
  author = {Matthew Cockayne},
  title = {MedXAI: Interactive Medical Imaging Explainability Interface},
  year = {2025},
  url = {https://github.com/Matt-Cockayne/MedXAI}
}
```

## References

- **LIME**: Ribeiro et al., "Why Should I Trust You?" (KDD 2016)
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NIPS 2017)
- **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
- **MedMNIST**: Yang et al., "MedMNIST v2" (Scientific Data 2023)
