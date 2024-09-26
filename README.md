# Pneumonia Detection Using Deep Learning Models

This project uses several deep learning architectures to classify chest X-ray images as either "Pneumonia" or "Normal." The models fine-tuned in this project include VGG16, ResNet50, EfficientNetB0, InceptionV3, and Xception.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Introduction
Pneumonia is one of the major causes of death worldwide, particularly in children and the elderly. Early detection can lead to better treatment outcomes. In this project, we leverage state-of-the-art CNN models pre-trained on ImageNet to detect pneumonia from chest X-ray images.

## Models
The following pre-trained models were used in this project and fine-tuned for pneumonia detection:
- **VGG16**
- **ResNet50**
- **EfficientNetB0**
- **InceptionV3**
- **Xception**

Each model was evaluated based on accuracy, precision, recall, F1 score, and AUC.

## Dataset
The dataset used in this project is the publicly available Chest X-ray dataset from Guangzhou Women and Children’s Medical Center. It contains 5,863 labeled images, categorized into two classes: "Pneumonia" and "Normal." The dataset is split into three sets:
- **Training set**: 5,216 images
- **Validation set**: 16 images
- **Test set**: 624 images

## Training
To fine-tune the models, various hyperparameters such as learning rate, optimizer, and batch size were used. The models were trained using a combination of data augmentation techniques, and early stopping was employed to prevent overfitting.

## Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: The proportion of correctly classified samples.
- **Precision**: The proportion of true positive samples among all positive predictions.
- **Recall**: The proportion of true positives correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.
- **AUC (Area Under Curve)**: The ROC curve was used to assess the models' performance.

## Results
The following table summarizes the performance metrics of each model:

| Model           | Accuracy | Precision | Recall  | F1 Score | AUC    |
|-----------------|----------|-----------|---------|----------|--------|
| VGG16           | 91.51%   | 0.9201    | 0.9846  | 0.9504   | 0.9788 |
| ResNet50        | 90.80%   | 0.8238    | 0.9949  | 0.9010   | 0.9697 |
| EfficientNetB0  | 86.91%   | 0.8651    | 0.8718  | 0.8685   | 0.9102 |
| InceptionV3     | 89.34%   | 0.8791    | 0.9347  | 0.9059   | 0.9635 |
| Xception        | 90.02%   | 0.8823    | 0.9554  | 0.9175   | 0.9685 |

## Future Work
- Explore more advanced models such as Vision Transformers (ViT) for better performance.
- Fine-tune the models with additional data and evaluate their robustness in different datasets.
- Incorporate explainability techniques (like Grad-CAM) to visualize the areas in the images the model is focusing on.

# Pneumonia Detection Using Deep Learning Models

This project uses several deep learning architectures to classify chest X-ray images as either "Pneumonia" or "Normal." The models fine-tuned in this project include VGG16, ResNet50, EfficientNetB0, InceptionV3, and Xception.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Introduction
Pneumonia is one of the major causes of death worldwide, particularly in children and the elderly. Early detection can lead to better treatment outcomes. In this project, we leverage state-of-the-art CNN models pre-trained on ImageNet to detect pneumonia from chest X-ray images.

## Models
The following pre-trained models were used in this project and fine-tuned for pneumonia detection:
- **VGG16**
- **ResNet50**
- **EfficientNetB0**
- **InceptionV3**
- **Xception**

Each model was evaluated based on accuracy, precision, recall, F1 score, and AUC.

## Dataset
The dataset used in this project is the publicly available Chest X-ray dataset from Guangzhou Women and Children’s Medical Center. It contains 5,863 labeled images, categorized into two classes: "Pneumonia" and "Normal." The dataset is split into three sets:
- **Training set**: 5,216 images
- **Validation set**: 16 images
- **Test set**: 624 images

## Training
To fine-tune the models, various hyperparameters such as learning rate, optimizer, and batch size were used. The models were trained using a combination of data augmentation techniques, and early stopping was employed to prevent overfitting.

## Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: The proportion of correctly classified samples.
- **Precision**: The proportion of true positive samples among all positive predictions.
- **Recall**: The proportion of true positives correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.
- **AUC (Area Under Curve)**: The ROC curve was used to assess the models' performance.

## Results
The following table summarizes the performance metrics of each model:

| Model           | Accuracy | Precision | Recall  | F1 Score | AUC    |
|-----------------|----------|-----------|---------|----------|--------|
| VGG16           | 91.51%   | 0.9201    | 0.9846  | 0.9504   | 0.9788 |
| ResNet50        | 90.80%   | 0.8238    | 0.9949  | 0.9010   | 0.9697 |
| EfficientNetB0  | 86.91%   | 0.8651    | 0.8718  | 0.8685   | 0.9102 |
| InceptionV3     | 89.34%   | 0.8791    | 0.9347  | 0.9059   | 0.9635 |
| Xception        | 90.02%   | 0.8823    | 0.9554  | 0.9175   | 0.9685 |

## Future Work
- Explore more advanced models such as Vision Transformers (ViT) for better performance.
- Fine-tune the models with additional data and evaluate their robustness in different datasets.
- Incorporate explainability techniques (like Grad-CAM) to visualize the areas in the images the model is focusing on.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

