# Histopathology Image Processing and TFRecord Generation (01_image_processing_tfrecord_generation.py)

## Overview

This Python script processes histopathology images categorized in subfolders based on class labels. It performs the following tasks:
1. **Image Collection**: Collects images from the specified dataset directory, where each subfolder represents a class.
2. **Duplicate Detection**: Detects and removes duplicate images using image hashing techniques.
3. **TFRecord Generation**: Converts the processed images into TFRecord format for efficient TensorFlow training.
4. **Stratified Cross-Validation**: Splits the dataset into stratified K-folds for training and validation, ensuring balanced class distribution.
5. **Data Visualization**: Generates plots to visualize the class distribution across the dataset and folds.

## Dataset

The dataset used in this paper is publicly available. Please download the relevant dataset from the corresponding source. You can find more information and download links for datasets like MHIST (Hyperplastic Polyp (HP) and Sessile Serrated Adenoma (SSA)) used in this example.

Once downloaded, structure the dataset as shown below.


## Requirements

Ensure the following dependencies are installed in your environment:

- Python 3.9+
- TensorFlow 2.10+
- TensorFlow Addons 0.18.0
- CUDA Toolkit 11.2+
- cuDNN 8.1+
- Additional Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
  - `Pillow`
  - `torch`
  - `imagehash`
  - `opencv-python`
Install the necessary dependencies.

Usage
## 1. Dataset Directory Structure
The dataset should be structured in the following way, with each subfolder representing a class:
/path/to/your/dataset/
    ├── Class1/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── Class2/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...

## For example, for the MHist dataset:
/g/data/nk53/mr3328/bracs/mhist/train/
    ├── Hyperplastic/
    ├── Sessile Adenoma/

## 2. Running the Script
Update the val_dir and new_dir paths in the script to point to your dataset:
val_dir = '/g/data/nk53/mr3328/bracs/mhist/train/'
new_dir = '/g/data/nk53/mr3328/bracs/mhist/train_images/'

Once the paths are updated, run the script as follows:

python image_processing_tfrecord_generation.py

## 3. Outputs
# train.csv: A CSV file containing image filenames and their corresponding labels.
# TFRecords: Serialized TFRecord files for each fold, which can be used for TensorFlow training.
# Sample Images: A set of saved sample images for validation purposes.
# Class Distribution Plots: Bar plots showing the class distribution across the entire dataset and each fold.


**--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# Training EfficientNet-B3 with Contrastive Learning for Histopathology Image Classification

The script `02_train_efficientnet_b3_contrastive.py` is designed to train an EfficientNet-B3 model with contrastive learning.

## Overview
This project demonstrates how to train an EfficientNet-B3 model for binary classification (Hyperplastic Polyp (HP) and Sessile Serrated Adenoma (SSA)) using contrastive learning on histopathology images. The workflow involves pre-training an encoder with supervised contrastive loss and then fine-tuning a classifier with a frozen encoder. It also implements cross-validation and generates various visualizations such as confusion matrices and embedding visualizations using TSNE.

## Key Features
1. **Supervised Contrastive Learning**: Pre-trains the encoder using supervised contrastive loss.
2. **EfficientNet-B3 Encoder**: Leverages a pre-trained EfficientNet-B3 as the encoder model.
3. **Data Augmentation**: A variety of augmentations (e.g., flipping, rotation, shearing, CutOut) are applied to the dataset to enhance generalization.
4. **Cosine Learning Rate Scheduling**: A custom learning rate schedule with warmup for efficient training.
5. **Cross-Validation**: Implements K-fold cross-validation to train and evaluate the model.
6. **Performance Metrics**: Provides precision, recall, F1-score, and accuracy for each fold, along with an overall summary.
7. **Embedding Visualization**: Uses TSNE to visualize embeddings from the trained model.
8. **Training & Validation Metrics Visualization**: Plots accuracy and loss across all folds for better understanding of model performance.

## Training EfficientNet-B3 for Histopathology Image Classification

The script `02_train_efficientnet_b3_histopathology.py` is designed to perform the following tasks:

1. **Data Augmentation**: Apply a variety of image transformations (rotation, flipping, shearing, CutOut, etc.) to augment the dataset during training.
2. **Model Training**: Train an EfficientNet-B3 model for binary classification (HP and SSA classes).
3. **Cosine Learning Rate Schedule**: The script implements a custom cosine learning rate scheduler with warmup to dynamically adjust the learning rate during training.
4. **Cross-Validation**: Uses K-fold cross-validation to train and validate the model across different folds of the dataset.
5. **Performance Visualization**: After training, the script generates various visualizations such as training/validation accuracy and loss, confusion matrices, and TSNE plots for embedding visualization.
6. **Metrics Calculation**: It calculates and prints out precision, recall, F1-score, and accuracy for each fold and provides an overall performance summary across all folds.


---

# Training ResNet50 with Contrastive Learning for Histopathology Image Classification

The script `02_train_resnet50_contrastive.py` is designed to train an ResNet50 model with contrastive learning.

## Overview
This project demonstrates how to train an ResNet50 model for binary classification (Hyperplastic Polyp (HP) and Sessile Serrated Adenoma (SSA)) using contrastive learning on histopathology images. The workflow involves pre-training an encoder with supervised contrastive loss and then fine-tuning a classifier with a frozen encoder. It also implements cross-validation and generates various visualizations such as confusion matrices and embedding visualizations using TSNE.

## Key Features
1. **Supervised Contrastive Learning**: Pre-trains the encoder using supervised contrastive loss.
2. **ResNet50 Encoder**: Leverages a pre-trained ResNet50 as the encoder model.
3. **Data Augmentation**: A variety of augmentations (e.g., flipping, rotation, shearing, CutOut) are applied to the dataset to enhance generalization.
4. **Cosine Learning Rate Scheduling**: A custom learning rate schedule with warmup for efficient training.
5. **Cross-Validation**: Implements K-fold cross-validation to train and evaluate the model.
6. **Performance Metrics**: Provides precision, recall, F1-score, and accuracy for each fold, along with an overall summary.
7. **Embedding Visualization**: Uses TSNE to visualize embeddings from the trained model.
8. **Training & Validation Metrics Visualization**: Plots accuracy and loss across all folds for better understanding of model performance.

## Training ResNet50 for Histopathology Image Classification

The script `02_train_resnet50_histopathology.py` is designed to perform the following tasks:

1. **Data Augmentation**: Apply a variety of image transformations (rotation, flipping, shearing, CutOut, etc.) to augment the dataset during training.
2. **Model Training**: Train an ResNet50 model for binary classification (HP and SSA classes).
3. **Cosine Learning Rate Schedule**: The script implements a custom cosine learning rate scheduler with warmup to dynamically adjust the learning rate during training.
4. **Cross-Validation**: Uses K-fold cross-validation to train and validate the model across different folds of the dataset.
5. **Performance Visualization**: After training, the script generates various visualizations such as training/validation accuracy and loss, confusion matrices, and TSNE plots for embedding visualization.
6. **Metrics Calculation**: It calculates and prints out precision, recall, F1-score, and accuracy for each fold and provides an overall performance summary across all folds.

---


# Training EfficientNet-B3 and ResNet50 for Histopathology Image Classification with Supervised Contrastive Learning

The script `02_train_efficientnet_b3_resnet50_contrastive.py` is designed to train combined EfficientNet-B3 and ResNet50 model with contrastive learning.

## Overview

This project demonstrates how to combine EfficientNet-B3 and ResNet50 for histopathology image classification using a supervised contrastive learning approach. The model is trained to classify between two types of polyps: Hyperplastic Polyp (HP) and Sessile Serrated Adenoma (SSA). The workflow includes pre-training the encoders using a supervised contrastive loss and then fine-tuning a classifier with a frozen encoder.

### Key Features:
- **EfficientNet-B3 and ResNet50 Encoders**: The model uses both EfficientNet-B3 and ResNet50 pre-trained on ImageNet for feature extraction.
- **Supervised Contrastive Learning**: The encoder is pre-trained with a supervised contrastive loss to improve the quality of the learned embeddings.
- **Cross-Validation**: Uses K-Fold cross-validation (N=5) to evaluate the model's performance across different folds.
- **Data Augmentation**: Includes various data augmentation techniques (e.g., random cropping, shear, rotation, CutOut) to improve generalization.
- **Cosine Learning Rate Scheduling**: A custom learning rate scheduler with warmup and optional max learning rate holding is used for efficient training.
- **Embedding Visualization**: Uses TSNE to visualize the learned embeddings.
- **Performance Visualization**: Plots training/validation accuracy, loss, and confusion matrices for all folds.


## Outputs
- **Model Weights**: Saved for each fold (e.g., `model_fold_1.h5`).
- **Performance Metrics**: Precision, recall, F1-score, and accuracy across folds.
- **Embedding Plots**: TSNE visualizations of the embeddings.
- **Confusion Matrices**: Saved confusion matrix plots for each fold.
