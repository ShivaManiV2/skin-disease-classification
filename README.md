Skin Disease Classification & Lesion Segmentation using Deep Learning
📌 Overview

This project presents a deep learning pipeline for automatic skin disease classification and lesion segmentation using the HAM10000 dataset. The system combines transfer learning-based classification models with medical image segmentation networks to assist dermatological analysis.

The project compares multiple deep learning architectures and evaluates their performance on dermoscopic images.

 Objectives

Classify skin lesions into 7 disease categories

Perform lesion segmentation to identify affected regions

Compare lightweight vs deep architectures

Analyze class imbalance impact

Evaluate segmentation accuracy under brightness variation

🗂 Dataset

The project uses the HAM10000 Dataset.

Dataset information:

Total images: 10,015

Dermoscopic skin lesion images

7 diagnostic categories:

Class	Description
akiec	Actinic Keratoses
bcc	Basal Cell Carcinoma
bkl	Benign Keratosis
df	Dermatofibroma
mel	Melanoma
nv	Melanocytic Nevus
vasc	Vascular Lesions

⚠ The dataset is highly imbalanced, with the nevus (nv) class dominating the dataset.

To address this, techniques such as:

Weighted loss

Balanced sampling

Data augmentation

were applied.

 Models Implemented
1️ Classification Models
MobileNetV2

Lightweight architecture

Depthwise separable convolutions

Efficient for resource-constrained systems

Accuracy:
70%

ResNet18

Residual deep neural network

Better feature extraction

Stronger performance on medical images

Accuracy:
76%

2️ Segmentation Models
U-Net

Encoder–decoder architecture with skip connections designed for biomedical image segmentation.

Attention U-Net

Enhanced U-Net architecture with attention gates that allow the network to focus on relevant lesion regions.

Results
Classification Accuracy
Model	Accuracy
MobileNetV2	70%
ResNet18	76%

ResNet18 demonstrated better performance due to its deeper architecture.

Segmentation Performance
Condition	Dice	IoU	Precision	Recall
Normal	0.885	0.818	0.918	0.881
Brightness Variation	0.891	0.820	0.885	0.924

The segmentation model maintained high performance even under brightness changes.

Visualizations

The project generates multiple plots for analysis:

Dataset class distribution

Model accuracy comparison

Per-class prediction comparison

Confusion matrices

Segmentation overlays

Example overlay visualization:

🟡 Correct overlap

🟢 Missed lesion area

🔴 Over-segmentation

⚙️ Tech Stack

Programming Language

Python

Deep Learning

PyTorch

Torchvision

Data Processing

NumPy

Pandas

Visualization

Matplotlib

Seaborn

Evaluation

Scikit-learn

Web Interface (optional)

Streamlit

Project Structure

skin-disease-classification/
│
├── classification/              # Classification models and scripts
│   ├── main.py
│   ├── main2.py
│   ├── model_comparision.py
│   └── pre_class_corr_prediction.py
│
├── segmentation/                # Segmentation models and scripts
│   ├── segmentation.py
│   ├── segmentation2.py
│   ├── compare_segmentation.py
│   ├── overlay.py
│   └── overlay_all.py
│
├── data/                        # Dataset (Images, Masks, Metadata)
│   ├── images/
│   ├── masks/
│   └── HAM10000_metadata.csv
│
├── app.py                       # Streamlit Web Interface
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── SHIVA_Skin_Disease_report.pdf # Project report

Installation

1. Clone the repository:
```bash
git clone https://github.com/ShivaManiV2/skin-disease-classification.git
cd skin-disease-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Running the Project

Train Classification Model
```bash
python classification/main.py
```

Train Segmentation Model
```bash
python segmentation/segmentation.py
```

Run Streamlit Interface
```bash
streamlit run app.py
```
References

Tschandl, P., Rosendahl, C., & Kittler, H.
The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific Data, 2018.

He, K., Zhang, X., Ren, S., & Sun, J.
Deep Residual Learning for Image Recognition.
CVPR 2016.

Sandler, M. et al.
MobileNetV2: Inverted Residuals and Linear Bottlenecks.
CVPR 2018.

Ronneberger, O., Fischer, P., & Brox, T.
U-Net: Convolutional Networks for Biomedical Image Segmentation.
MICCAI 2015.

Oktay, O. et al.
Attention U-Net: Learning Where to Look for the Pancreas.
2018.

Author

Developed as part of a Data Science / Computer Vision academic project.
