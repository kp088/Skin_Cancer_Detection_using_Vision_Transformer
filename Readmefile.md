# 🧠 AI700 Project – Skin Lesion Classification using Vision Transformer (ViT)

 **Course:** AI700-001 (Fall 2025) – Deep learning
 
 **University:** Long Island University – Brooklyn Campus  
 
 **Team Members:**  
  Rashmi Thimmaraju  
 Binh Diep  
 Kartavya Mandora  
 Kirtan Patel  
 
**Date:** December 12, 2025  

 **Instructor:** Prof. Reda Nacif ElAlaoui (Course Instructor)

---

## 📄 Overview

Skin cancer is the **most common cancer in the United States**, affecting 1 in 5 Americans by age 70.  
Early diagnosis and treatment significantly
improve survival rates, especially for **melanoma**, the most aggressive form.

This project implements a **Vision Transformer (ViT)** 
architecture to automatically detect and classify skin lesions from dermatoscopic images.  
Our goal is to support **AI-assisted clinical decision systems** 
that help dermatologists achieve faster and more accurate diagnoses.

---

## 💡 Application

AI-powered early detection tools can:
- Reduce clinical diagnostic time  
- Improve diagnostic accuracy for melanoma and other lesions  
- Support healthcare professionals in identifying malignancies at early stages  

---

## 📚 Literature Review

| Study | Approach | Accuracy |
|-------|-----------|-----------|
| **Aldealnabi et al. (2024)** | Vision Transformer for global attention & local context learning | 96% |
| **Bhimavarapu et al. (2022)** | Fuzzy GC-SCNN using segmentation & fuzzy logic | 99.75% |
| **Al-Waisy et al. (2025)** | Deep hybrid model (Mask-R-CNN + GrabCut + HRNet + attention) | 100% |

👉 **Observation:** Vision Transformers outperform traditional CNNs by capturing both global and local features.

---

## 🧩 Dataset – HAM10000 (Kaggle)

**Source:** [HAM10000 – Human Against Machine with 10,000 training images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

- **Images:** 10,015 dermatoscopic images  
- **Classes:** 7 skin lesion types (benign and malignant)
- **Format:** JPEG + metadata CSV (dx, sex, localization, age)
- **Preprocessing:** Resized to 224×224, normalized, augmented (flips, contrast, rotations)

| Label | Meaning | Type |
|--------|----------|------|
| nv | Melanocytic nevi | Benign |
| mel | Melanoma | Malignant |
| bkl | Benign keratosis | Benign |
| bcc | Basal Cell Carcinoma | Malignant |
| akiec | Actinic Keratoses | Malignant |
| vasc | Vascular lesions | Benign |
| df | Dermatofibroma | Benign |

---

## ⚙️ Methodology

**Step-by-Step Pipeline**

1. **Dataset Collection:** Import HAM10000 dataset (images + metadata).  
2. **Preprocessing:** Resize, normalize, balance, augment images.  
3. **Feature Extraction:** Vision Transformer base (patch16-224).  
4. **Training:** Backpropagation using AdamW optimizer.  
5. **Evaluation:** Compute accuracy, F1, precision, recall, confusion matrix.  
6. **Comparison:** Evaluate ViT vs CNN vs prior literature.

---

## 🧠 Model Details

| Component | Description |
|------------|-------------|
| **Architecture** | Vision Transformer (ViT-Base-Patch16-224, pretrained on ImageNet) |
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | AdamW (LR = 3e-5) |
| **Batch Size** | 32 |
| **Epochs** | 20 |
| **Frameworks** | PyTorch, Timm, Albumentations |

---

## 🚀 Run Instructions

1. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
Download dataset

pip install kaggle

kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000/


Train model

python transformer_model.py


Evaluate

Confusion matrix and classification report auto-generated
Model checkpoints saved in outputs/best_model.pth

📊 Results and Evaluation

✅ CNN Model:

Accuracy: 70.53%
Loss: ≈1.20


✅ Vision Transformer (ViT) Model:

Metric	Value
Training Accuracy	96%
Testing Accuracy	89%
Macro Avg F1-Score	0.82
Weighted Avg F1-Score	0.89


🧾 Observation:

ViT outperformed CNN models, 
showing improved precision and recall for minority lesion classes.


📈 Model Comparison

Aspect	CNN	ViT
Feature Learning	Convolutional filters	Attention-based transformer
Accuracy	70.53%	96%
Data Size Handling	Small	Large & Complex
Effectiveness	Simple image tasks	Complex global feature extraction
Loss	1.20	0.03

🎯 Key Achievements

Achieved 96% accuracy using Vision Transformer (ViT).
Model effectively captures both local details and global patterns.
Outperformed traditional CNN approaches.
Strong potential for early and accurate melanoma detection.
Supports development of AI-based clinical diagnostic systems.

🔮 Future Work

Integrate AGCWD (Adaptive Contrast Enhancement) and hybrid Mask R-CNN + GrabCut segmentation.
Validate model robustness on larger datasets (ISIC 2019, PH2).
Use Grad-CAM and attention heatmaps for explainability.
Incorporate clinical metadata (patient history, lesion evolution).
Develop web-based diagnostic tool for clinician use.

👩‍💻 Authors

Rashmi Thimmaraju	-->ViT model implementation, training & evaluation

Binh Diep	-->	Model comparison & documentation

Kartavya Mandora	--->Data preprocessing & augmentation

Kirtan Patel ----->Report & presentation preparation

🪪 License

This project is released under the Apache 2.0 License

Dataset © Kaggle – used under CC BY-NC.
