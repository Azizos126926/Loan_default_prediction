# **Load Default Prediction using Machine Learning** 📊  

## **Overview**  
This project focuses on predicting loan defaults using machine learning techniques. It leverages multiple classification models to determine the likelihood of a borrower defaulting based on various financial and personal attributes.  

## **Features**  
✔️ **Data Preprocessing**: Handling missing values, encoding categorical features, and feature scaling  
✔️ **Class Imbalance Handling**: SMOTE (oversampling), undersampling techniques  
✔️ **Model Training & Evaluation**: Logistic Regression, XGBoost, LightGBM, Random Forest, and Stacking Classifier  
✔️ **Performance Metrics**: Accuracy, Precision, Recall, F1-score  
✔️ **Result Visualization**: Comparison of model performance  

## **Dataset**  
The dataset contains historical loan records with features such as:  
- **Demographic Information**: Age, employment status, income level  
- **Financial Attributes**: Loan amount, credit history, repayment status  
- **Target Variable**: Defaulted (`1`) or Not Defaulted (`0`)  


This project consists of two Jupyter Notebooks:
1. **Exploratory Data Analysis (EDA)**: A detailed analysis of the dataset to understand its structure, patterns, and potential issues.
2. **Model Training**: A notebook focused on training and evaluating machine learning models using the dataset.

## Notebooks

### EDA Notebook
- **File**: `01_eda.ipynb`
- **Purpose**: This notebook performs an in-depth exploratory data analysis (EDA) on the dataset. It includes:
  - Loading and inspecting the dataset.
  - Handling missing values and outliers.
  - Visualizing data distributions and relationships.
  - Feature engineering and data preprocessing.

### Model Training Notebook
- **File**: `02_model_training.ipynb`
- **Purpose**: This notebook focuses on training and evaluating machine learning models. It includes:
  - Loading the preprocessed dataset.
  - Splitting the data into training and testing sets.
  - Training multiple models (e.g., Logistic Regression, XGBoost, Random Forest, etc.).
  - Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
  - Addressing class imbalance using techniques like SMOTE and undersampling.
  - Hyperparameter tuning using GridSearchCV.
  - Comparing model performance using visualizations.

## **Usage**  
Run the model training script:  
```bash
python train_model.py
```
or open and execute the Jupyter Notebook:  
```bash
jupyter notebook 02_model_training.ipynb
```

## **Model Performance**  
📊 **Best Model**: **Stacking Classifier** (Highest accuracy & F1-score)  
📉 **Baseline Model**: Logistic Regression (Lower recall, sensitive to class imbalance)  

### **Performance Comparison**  
| Model | Accuracy | Precision | Recall | F1-Score |  
|--------|---------|-----------|--------|---------|  
| Logistic Regression | 85% | 55% | 10% | 17% |  
| Logistic Regression (SMOTE) | 72% | 23% | 68% | 34% |  
| Logistic Regression (Undersampling) | 70% | 22% | 65% | 33% |  
| XGBoost | 87% | 57% | 12% | 19% |  
| Random Forest | 83% | 48% | 15% | 23% |  
| LightGBM | 88% | 58% | 13% | 21% |  
| **Stacking Classifier** | **89%** | **60%** | **14%** | **23%** |  

## **Results Visualization**  
📉 Below is a comparison of different models based on Accuracy, Precision, Recall, and F1-Score:  
![Model Comparison](results/model_comparison.png)  




