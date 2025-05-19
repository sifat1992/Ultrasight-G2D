# Learnable 2D Gaussian Filters for Computationally Efficient Abdominal Organ Classification

This project explores the integration of learnable 2D Gaussian filters with standard and custom CNN architectures to improve the classification of abdominal ultrasound images. The work focuses on enhancing computational efficiency and performance in medical imaging classification tasks.

## 🚀 Key Features
- Implementation of a learnable 2D Gaussian filter layer.
- Comparative analysis using:
  - DenseNet121
  - AlexNet
  - ResNet50
  - VGG16
  - Custom CNN
  - Custom CNN + Learnable 2D Gaussian Filter
- Target task: Abdominal organ classification from ultrasound images.
- Optimized for performance and interpretability.

## 📄 Paper

> [Download Full Paper (PDF)](Results/SPIE2025_GaussFilters_Shaila.pdf)  
> *"Learnable 2D Gaussian Filters for Computationally Efficient Abdominal Organ Classification"*  
> Presented at SPIE 2025 Conference on Real-Time Image Processing and Deep Learning  
> *(Manuscript not yet published — PDF uploaded for reference only)*

## 🧪 Dataset
- Custom curated ultrasound dataset (preprocessed and balanced).
- Not publicly available due to sensitivity and privacy constraints.
- Contact for access if needed for research collaboration.
- ## 📸 Sample Images

### Healthy Image Example
![Healthy Sample](assets/healthy_example1.png)

### Abnormal Image Example
![Abnormal Sample](assets/abnormal_example1.png)

📊 Results – Model Performance Summary
🧠 AlexNet
Accuracy Log
Classification Report

🧠 Custom CNN
Accuracy Log
Classification Report

🧠 DenseNet121
Accuracy Log
Classification Report

🧠 GaussNet (Custom CNN + Learnable 2D Gaussian)
Accuracy Log
Classification Report

🧠 ResNet50
Accuracy Log
Classification Report

🧠 VGG16
Accuracy Log
Classification Report
## 📁 Project Structure

```
├── assets/                         # Folder containing sample dataset images
│   ├── healthy_example1.png
│   ├── abnormal_example1.png
├── Results                         # Folder containing images of results
│   ├── alexnet_acc.png
│   ├── alexnet_class.png
│   ├── custom_cnn_acc
|   ├── custom_cnn_class
|   ├── dense_acc
|   ├── dense_acc
|   ├── gauss_acc
|   ├── gauss_class
|   ├── res_acc
|   ├── res_class
|   ├── vgg_acc
|   ├── vgg_class
├── alexnet.py                      # Training and evaluation using AlexNet
├── dense-121.py                    # Training and evaluation using DenseNet121
├── resnet50_paper_fif.py           # Training and evaluation using ResNet50
├── vgg_16_PAPER_fif.py             # Training and evaluation using VGG16
├── custom_cnn_paper_fif.py         # Training and evaluation of custom CNN
├── gauss_final.py                  # Custom CNN with Learnable 2D Gaussian layer
├── gaussiand2D_layer_pytorch.py    # Script defining the learnable 2D Gaussian layer
├── create_dataset.py               # Dataset loading and preprocessing
└── requirements.txt/               # Required Python dependencies
└── .gitignore                      # Specifies files and folders to be ignored by Git
├── README.md                       # Reading this!
```

References:
```
[1] G. Huang, Z. Liu, L. Van Der Maaten, and K. Weinberger, “Densely Connected Convolutional Networks,” Jan. 2018. Available: https://arxiv.org/pdf/1608.06993
[2] K. Simonyan and A. Zisserman, “VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION,” Apr. 2015. Available: https://arxiv.org/pdf/1409.1556
[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv.org, Dec. 10, 2015. https://arxiv.org/abs/1512.03385
[4] S. Biswas, Cemre Omer Ayna, and A. C. Gurbuz, “PLFNets: Interpretable Complex Valued Parameterized Learnable Filters for Computationally Efficient RF Classification,” IEEE Transactions on Radar Systems, pp. 1–1, Jan. 2024, doi: https://doi.org/10.1109/trs.2024.3486183.
[5} “Papers with Code - ImageNet Classification with Deep Convolutional Neural Networks,” paperswithcode.com. https://paperswithcode.com/paper/imagenet-classification-with-deep
[6] A. Persson, “Aladdin Persson,” YouTube. https://www.youtube.com/@AladdinPersson (accessed May 19, 2025).
```

