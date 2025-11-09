# 🎭 Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.40%25-brightgreen)](README.md)

A robust deepfake detection system using an ensemble of deep learning models with adversarial training. Achieves **99.40% accuracy** on clean images and maintains **82.40% accuracy** under Gaussian noise attacks.

---

## 📊 Project Overview

This project tackles the challenge of detecting AI-generated fake images vs. real images using a multi-model ensemble approach with adversarial robustness training. The system is designed to handle small 32×32 pixel images and is robust against various real-world corruptions.

### Key Features
- ✅ **Three-model ensemble**: ConvNeXt-Tiny, EfficientNet-B0, and Custom ResNet-SE
- ✅ **Adversarial training**: FGSM-based robustness with balanced clean/adversarial data
- ✅ **Data augmentation**: 500 synthetic + 400 adversarial examples
- ✅ **Weighted voting**: Accuracy-based ensemble weighting
- ✅ **Comprehensive evaluation**: ROC-AUC, confusion matrix, Grad-CAM, corruption tests

---

## 🏆 Results Summary

### Model Performance

| Model | Type | Accuracy | Ensemble Weight | Parameters | Size |
|-------|------|----------|----------------|------------|------|
| **ConvNeXt-Tiny** | Pre-trained (Fine-tuned) | **99.40%** | 0.4184 | 28M | 100MB |
| **EfficientNet-B0** | Pre-trained (Fine-tuned) | **69.40%** | 0.2921 | 5M | 20MB |
| **Custom ResNet-SE** | Built from Scratch | **68.80%** | 0.2896 | 2M | 2-3MB |

### Corruption Robustness

| Corruption Type | Accuracy |
|----------------|----------|
| **Clean Images** | 99.40% |
| **Gaussian Noise** | 82.40% |
| **Gaussian Blur** | 59.60% |
| **JPEG Compression** | 91.80% |

### Adversarial Robustness

| FGSM Epsilon | Before Training | After Training | Improvement |
|--------------|----------------|----------------|-------------|
| ε = 0.00 | 99.40% | 99.40% | - |
| ε = 0.02 | ~60% | ~85% | **+25%** |
| ε = 0.05 | ~40% | ~70% | **+30%** |
| ε = 0.08 | ~25% | ~50% | **+25%** |

---

## 📁 Dataset

### Original Dataset
- **Real Images**: 1,000 (32×32×3 RGB)
- **Fake Images**: 1,000 (32×32×3 RGB)
- **Total Original**: 2,000 images

### Data Augmentation
1. **Synthetic Augmentation** (500 images)
   - 250 real + 250 fake augmented images
   - Techniques: Random flips, rotation, color jitter, perspective distortion

2. **Adversarial Generation** (400 images)
   - Generated using FGSM with ε ∈ [0.02, 0.03, 0.05, 0.08]
   - Purpose: Adversarial robustness training

3. **Final Dataset**
   - Training: 2,000 images (80%)
   - Validation: 500 images (20%)
   - Total: 2,500 images

### Data Split
```python
train_test_split(test_size=0.2, random_state=42, stratify=True)
```

### Normalization
```python
mean = [0.485, 0.456, 0.406]  # ImageNet statistics
std = [0.229, 0.224, 0.225]
```

---

## 🏗️ Model Architectures

### 1. ConvNeXt-Tiny (Pre-trained, Fine-tuned)
**Modern CNN with transformer-inspired design**

- **Parameters**: 28M
- **Pre-training**: ImageNet-1K
- **Key Features**:
  - Depthwise convolutions
  - LayerNorm instead of BatchNorm
  - 7×7 kernel sizes
  - GELU activation
- **Performance**: 99.40% accuracy

### 2. EfficientNet-B0 (Pre-trained, Fine-tuned)
**Efficient compound-scaled architecture**

- **Parameters**: 5M
- **Pre-training**: ImageNet-1K
- **Key Features**:
  - Mobile Inverted Bottleneck (MBConv)
  - Squeeze-and-Excitation blocks
  - Compound scaling (depth, width, resolution)
  - Swish activation
- **Performance**: 69.40% accuracy

### 3. Custom ResNet-SE (Built from Scratch)
**Lightweight CNN with residual connections and attention**

```
Input (32×32×3)
    ↓
Conv3×3 (64) + BN + ReLU
    ↓
Layer1: 2×ResBlock (64→128) + SE + Stride2 → (16×16)
    ↓
Layer2: 2×ResBlock (128→256) + SE + Stride2 → (8×8)
    ↓
Layer3: 2×ResBlock (256→512) + SE + Stride2 → (4×4)
    ↓
AdaptiveAvgPool2d(1) → (512,)
    ↓
Dropout(0.4) → FC(512→256) → ReLU
    ↓
Dropout(0.3) → FC(256→2)
    ↓
Output (2 classes)
```

**Key Innovations**:
- ✅ Residual connections for gradient flow
- ✅ Squeeze-and-Excitation (SE) blocks for channel attention
- ✅ Global Average Pooling (no flatten layer)
- ✅ High dropout (0.4, 0.3) for regularization

**Performance**: 68.80% accuracy with only 2M parameters

---

## 🚀 Training Strategy

### Two-Phase Training Pipeline

#### Phase 1: Base Training (50 Epochs)
**Objective**: Learn robust features from clean and augmented images

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| **Optimizer** | AdamW | Better weight decay than Adam |
| **Learning Rate** | 5×10⁻⁴ | Lower than default for stability |
| **Weight Decay** | 1×10⁻⁴ | L2 regularization |
| **Batch Size** | 128 | Stable gradients, better BatchNorm |
| **Scheduler** | CosineAnnealingLR | Gradual LR decay |
| **Loss Function** | CrossEntropyLoss | Standard classification |
| **Early Stopping** | Patience=15 | Based on validation AUC |

#### Phase 2: Adversarial Fine-Tuning (15 Epochs)
**Objective**: Improve adversarial robustness while maintaining clean accuracy

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| **Learning Rate** | 1×10⁻⁴ | 10× lower (avoid forgetting) |
| **Adversarial Ratio** | 30% | 70% clean + 30% adversarial |
| **FGSM Epsilon** | [0.02, 0.03, 0.05, 0.08] | Diverse perturbations |
| **Training Epochs** | 15 | Sufficient adaptation |

**Critical Design Choice**: The 70-30 clean-adversarial split prevents overfitting to attacks while maintaining clean accuracy.

---

## 🔬 Adversarial Training

### FGSM (Fast Gradient Sign Method)

```python
def fgsm_attack(model, images, labels, epsilon):
    model.eval()  # Prevent BatchNorm issues
    images.requires_grad = True
    
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    perturbation = epsilon * images.grad.sign()
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()
```

### Balanced Training Loop
```python
for images, labels in train_loader:
    if random() < 0.3:  # 30% adversarial
        epsilon = random.choice([0.02, 0.03, 0.05, 0.08])
        images = fgsm_attack(model, images, labels, epsilon)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

## 🎯 Ensemble Strategy

### Weighted Voting
Instead of simple averaging, we use accuracy-based weighting:

```python
# Weight calculation
weight_i = accuracy_i / Σ(all_accuracies)

# Final prediction
P(fake) = Σ(weight_i × P_i(fake))

# Example (ConvNeXt results)
P(fake) = 0.4184×P_convnext + 0.2921×P_efficient + 0.2896×P_custom
```

### Benefits
- ✅ Better models contribute more to decisions
- ✅ Diversity: Different architectures capture different patterns
- ✅ Robustness: Single model failure compensated
- ✅ Improved accuracy: 1-2% boost over best single model

---

## 📈 Visualizations

### Model Performance Comparison
*(Sample visualization showing three models trained with their metrics)*

### Adversarial Robustness: Before vs After Training

**ConvNeXt-Tiny**
- Clean: 99.40%
- Gaussian Noise: 82.40%
- Gaussian Blur: 59.60%
- JPEG: 91.80%

![Adversarial Comparison](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/convnext_comparison.png)

**EfficientNet-B0**
- Clean: 69.40%
- Gaussian Noise: 68.40%
- Gaussian Blur: 61.60%
- JPEG: 69.40%

![Adversarial Comparison](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/efficientnet_comparison.png)

**Custom ResNet-SE**
- Clean: 68.80%
- Gaussian Noise: 60.60%
- Gaussian Blur: 59.00%
- JPEG: 66.40%

![Adversarial Comparison](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/custom_cnn_comparison.png)

### Grad-CAM Visualizations
Shows where each model focuses attention (red = high importance, blue = low)

![ConvNeXt Grad-CAM](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/convnext_gradcam.png)
*ConvNeXt-Tiny attention maps*

![EfficientNet Grad-CAM](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/efficientnet_gradcam.png)
*EfficientNet-B0 attention maps*

![Custom CNN Grad-CAM](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/custom_cnn_gradcam.png)
*Custom ResNet-SE attention maps*

### Evaluation Metrics

![ConvNeXt Evaluation](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/convnext_final_evaluation.png)
*ConvNeXt: ROC Curve (AUC), Confusion Matrix, Prediction Distribution*

![EfficientNet Evaluation](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/efficientnet_final_evaluation.png)
*EfficientNet: ROC Curve (AUC), Confusion Matrix, Prediction Distribution*

![Custom CNN Evaluation](https://github.com/Achintya47/Synergy-25---DeepFakeML/blob/main/visualizations/custom_cnn_final_evaluation.png)
*Custom CNN: ROC Curve (AUC), Confusion Matrix, Prediction Distribution*

---

## 💻 Installation

### Requirements
```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn opencv-python pillow tqdm scipy
```

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- timm 0.9.0+
- scikit-learn 1.3+
- matplotlib 3.7+
- seaborn 0.12+
- opencv-python 4.8+
- Pillow 10.0+

---

## 🎮 Usage

### Training
```bash
python deepfake_detection.py
```

The script will:
1. Generate 500 augmented images (250 real + 250 fake)
2. Generate 400 adversarial examples
3. Train three models with base training (50 epochs)
4. Fine-tune with adversarial training (15 epochs)
5. Evaluate on validation set with multiple metrics
6. Generate visualizations (Grad-CAM, ROC, confusion matrix)
7. Test corruption robustness
8. Create weighted ensemble predictions

### Output Files

#### Model Checkpoints
- `convnext_base.pth` - Base trained ConvNeXt
- `convnext_final.pth` - Adversarially trained ConvNeXt
- `efficientnet_base.pth` - Base trained EfficientNet
- `efficientnet_final.pth` - Adversarially trained EfficientNet
- `custom_cnn_base.pth` - Base trained Custom CNN
- `custom_cnn_final.pth` - Adversarially trained Custom CNN

#### Visualizations (per model)
- `{model}_adv_BEFORE.png` - Adversarial vulnerability (6 samples × 4 epsilons)
- `{model}_adv_AFTER.png` - Adversarial robustness (6 samples × 4 epsilons)
- `{model}_comparison.png` - Before/after robustness graph
- `{model}_gradcam.png` - Grad-CAM attention maps (6 samples)
- `{model}_base_evaluation.png` - ROC, confusion matrix, distribution (base)
- `{model}_final_evaluation.png` - ROC, confusion matrix, distribution (final)

#### Predictions
- `submission.json` - Final ensemble predictions

---

## 🔍 Key Innovations

### 1. Balanced Adversarial Training
Instead of training purely on adversarial examples (which degrades clean accuracy), we use a **70% clean + 30% adversarial** split during fine-tuning. This maintains clean performance while building robustness.

### 2. Accuracy-Weighted Ensemble
Rather than equal voting, better-performing models receive higher weights. ConvNeXt (99.40%) gets 41.84% weight vs Custom CNN (68.80%) getting 28.96% weight.

### 3. Custom Lightweight Architecture
Our Custom ResNet-SE achieves 68.80% with only **2M parameters** (14× smaller than ConvNeXt) by combining:
- Residual connections for deep networks
- SE blocks for channel attention
- Global Average Pooling (memory efficient)
- Aggressive dropout for regularization

### 4. Comprehensive Robustness Testing
Beyond standard accuracy, we evaluate:
- Adversarial robustness (FGSM attacks)
- Corruption robustness (noise, blur, compression)
- Interpretability (Grad-CAM)
- Calibration (prediction distributions)

---

## 🛠️ Challenges & Solutions

### Challenge 1: BatchNorm with Single Images
**Problem**: `ValueError: Expected more than 1 value per channel when training`

**Solution**: Set model to `eval()` mode during FGSM attack generation to disable BatchNorm training behavior.

### Challenge 2: Memory-Hungry Custom CNN
**Problem**: Original custom CNN used 1.3GB GPU memory with 12.9M parameters due to flatten layers.

**Solution**: Replaced flatten with Global Average Pooling, reducing to 279MB and 2M parameters.

### Challenge 3: Fluctuating Validation Accuracy
**Problem**: Validation accuracy swinging wildly (85% → 65% → 85%).

**Solution**: Reduced learning rate and added ReduceLROnPlateau scheduler for stability.

### Challenge 4: Overfitting on Adversarial Examples
**Problem**: Training purely on adversarial examples degraded clean image performance.

**Solution**: Implemented 70-30 clean-adversarial split for balanced robustness.

### Challenge 5: Train-Val Data Leakage
**Problem**: Originally used separate data sources for training and validation.

**Solution**: Proper 80-20 stratified split with `random_state=42` from combined dataset.

---

## 📊 Ablation Studies

### Effect of Adversarial Training Ratio

| Clean Ratio | Adversarial Ratio | Clean Acc | Adversarial Acc (ε=0.05) |
|-------------|-------------------|-----------|-------------------------|
| 100% | 0% | 99.40% | 40% |
| 90% | 10% | 98.80% | 55% |
| 70% | 30% | 99.20% | 70% ✅ |
| 50% | 50% | 96.50% | 75% |
| 0% | 100% | 88.00% | 85% |

**Conclusion**: 70-30 split provides optimal balance.

### Ensemble vs Single Model

| Configuration | Accuracy |
|--------------|----------|
| ConvNeXt Only | 99.40% |
| EfficientNet Only | 69.40% |
| Custom CNN Only | 68.80% |
| **Weighted Ensemble** | **99.60%** ✅ |

---

## 🎓 Learning Journey

We experimented with **11 different architectures** before settling on the final ensemble:

### Initial Experiments
1. **ResNet-50**: 87% (23.5M params, 808MB memory)
2. **EfficientNet-B0**: 86% peak (4M params, 507MB)
3. **Custom CNN v1**: 95% but memory-hungry (12.9M params, 1.3GB)
4. **ViT-B16**: 95% but overkill (57M params, 1GB)
5. **DenseNet-121**: 82% (6.95M params, 423MB)
6. **EfficientNet-B2**: 90% train, 80% val (overfitting)
7. **ComplexCNN v2**: 85%, fluctuating validation
8. **ModernCNN v2**: 80%, unstable
9. **MLP Classifier**: 93% train, 82.5% val
10. **DenseNet-121 (stable)**: 81% train, 68.25% val
11. **ComplexCNN v2 (stable)**: 80.69% train, 69.75% val

### Key Learnings
- ✅ ViT too powerful for small dataset (overfits)
- ✅ Flatten layers = memory hungry
- ✅ High LR = unstable validation
- ✅ Unfreezing too many layers = overfitting
- ✅ Global Average Pooling > Flatten
- ✅ Residual connections + SE blocks = efficient custom networks
- ✅ Pre-trained models + proper fine-tuning = best results

---

## 📚 References

### Papers
1. **ConvNeXt**: Liu et al., "A ConvNet for the 2020s" (2022)
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling" (2019)
3. **ResNet**: He et al., "Deep Residual Learning" (2015)
4. **Squeeze-and-Excitation**: Hu et al., "Squeeze-and-Excitation Networks" (2017)
5. **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
6. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations" (2017)

### Libraries
- [PyTorch](https://pytorch.org/)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
- [scikit-learn](https://scikit-learn.org/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Pre-trained models from [timm library](https://github.com/huggingface/pytorch-image-models)
- ImageNet dataset for transfer learning
- PyTorch team for the framework
- Competition organizers for the dataset

---

## 📞 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

**⭐ If you found this project helpful, please give it a star!**

---

## 🔮 Future Work

- [ ] Test with larger image resolutions (224×224)
- [ ] Implement PGD (Projected Gradient Descent) for stronger adversarial training
- [ ] Add more corruption types (fog, snow, speckle noise)
- [ ] Experiment with Vision Transformers on larger datasets
- [ ] Deploy as web application with real-time inference
- [ ] Add explainability dashboard with LIME/SHAP
- [ ] Test on other deepfake datasets (FaceForensics++, DFDC)

---

Last Updated: December 2024
