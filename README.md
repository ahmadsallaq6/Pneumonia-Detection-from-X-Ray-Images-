# Pneumonia Detection from X-Ray Images

This project aims to detect pneumonia from chest X-ray images using a deep learning model based on the ResNet50 architecture. The model was trained and fine-tuned on a dataset of pediatric chest X-rays, achieving notable performance in detecting pneumonia cases.

# Project Overview

Pneumonia is a serious respiratory infection, especially in children. Early detection of pneumonia is critical for proper medical care and can significantly improve patient outcomes. This model focuses on identifying pneumonia in pediatric chest X-ray images with high recall, ensuring the identification of true positive cases.

# Key Model Performance Metrics:

* Accuracy: 82.8%
* Recall: 93.33%
* Precision: 0.8235
* F1 Score: 0.8750

The model excels in predicting pneumonia cases, demonstrating strong capability in identifying true positives, which is crucial in medical diagnoses.

# Dataset

**Chest X-Ray Images (Pneumonia) dataset was used, containing:**

* 5,863 chest X-ray images (JPEG format)
* 2 categories: Pneumonia and Normal
* Images are organized into train, test, and validation folders
* Dataset link: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

# Dataset Details:

* Images were collected from pediatric patients (aged 1-5 years) at Guangzhou Women and Children’s Medical Center, China.
* The dataset consists of anterior-posterior chest X-ray images.
* Images were screened for quality, with two expert physicians grading the diagnoses, and a third expert checked the evaluation set for quality assurance.

# Model Architecture

The model was built using a pre-trained ResNet50 architecture, fine-tuned for binary classification (Pneumonia vs. Normal). The model was trained using the following techniques:

* Data Augmentation: To improve model robustness.
* Binary Crossentropy Loss: Optimized using the Adam optimizer.
* Recall focus: High priority was given to improving the recall to ensure accurate identification of pneumonia cases.

# Performance & Evaluation

While the model achieved a reasonable overall accuracy of 82.8%, it particularly excelled in identifying pneumonia cases, achieving a recall of 93.33%. This high recall indicates the model's strong ability to predict true positive cases, reducing the risk of missed diagnoses in pneumonia patients.

# Environment & Tools

Kaggle Notebook: The code was executed on Kaggle using their GPU accelerators for faster training and inference times.
Libraries used: TensorFlow, Keras, OpenCV, and others.

# Conclusion

This project provides an accurate and reliable model for detecting pneumonia in pediatric chest X-rays, with a strong focus on recall to ensure true positive identification. It can be further extended to include more diverse datasets and enhanced feature extraction techniques.
