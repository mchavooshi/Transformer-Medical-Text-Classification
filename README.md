# Transformer-Medical-Text-Classification

# Transformer Medical Text Classification

## Project Overview
This project focuses on comparing the performance of two transformer-based models, **Bio_ClinicalBERT** and **Bert-base-uncased**, in classifying medical texts. The dataset used is **123rc/medical_text** from Hugging Face, containing labeled text data split into five classes. The goal is to train, test, and evaluate these models to determine the better-performing one in this specific application.

## Models
- **[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)** from the `transformers` library by Hugging Face.
- **[Bert-base-uncased](https://huggingface.co/bert-base-uncased)** from the `transformers` library by Hugging Face.

## Dataset
- **[123rc/medical_text](https://huggingface.co/datasets/123rc/medical_text)**: A pre-labeled medical text dataset with five classes, sourced from Hugging Face, and already split into training and testing sets.

## Objectives
1. **Train and Test Models**: Train the **Bio_ClinicalBERT** and **Bert-base-uncased** models using the `transformers` library on the medical text dataset.
2. **Compare Performance**: Evaluate the models using metrics such as:
   - Confusion Matrix
   - Accuracy
   - Positive Predictive Value (PPV)
   - Negative Predictive Value (NPV)
3. **Determine Effectiveness**: Identify which model is more effective for medical text classification.

## Results
The project includes:
- Confusion matrices for both models.
- Accuracy, PPV, and NPV values.
- Analysis of model performance and suitability for medical text classification tasks.

## Installation
To run this project, ensure you have Python installed along with the necessary libraries:
```bash
pip install transformers datasets scikit-learn
