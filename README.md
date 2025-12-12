# Dataset Information

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset  
available on [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

📦 Dataset Details:
- ~10,000 dermatoscopic images
- 7 classes of pigmented skin lesions
- Format: JPEG images + CSV metadata

⚙️ To Download:
```bash
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000/

