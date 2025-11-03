# Automated MLOps Framework for Customer Churn Prediction

A comprehensive machine learning operations (MLOps) framework for predicting customer churn in the telecommunications industry. This project demonstrates end-to-end ML pipeline development including data preprocessing, model training, hyperparameter optimization, drift detection, and deployment with an interactive web interface.

## About

This portfolio project showcases practical MLOps skills by implementing an automated pipeline for customer churn prediction. The system trains multiple machine learning models, performs hyperparameter optimization, monitors for data drift, and provides an interactive interface for making predictions.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (A100 GPU, High RAM)

## Features

- **Automated Model Training**: Trains and compares XGBoost, LightGBM, and Random Forest models
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning (30 trials)
- **Model Versioning**: SQLite-based model registry with performance tracking
- **Data Drift Detection**: Kolmogorov-Smirnov statistical test for distribution changes
- **Feature Engineering**: Creates derived features including tenure groups, charge ratios, and service counts
- **Class Balancing**: SMOTE implementation to handle imbalanced dataset
- **Interactive Interface**: Gradio web application for predictions and system monitoring
- **Model Explainability**: Feature importance visualization
- **Performance Monitoring**: Tracks training time, inference latency, and cost metrics

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **License:** Database Contents License (DbCL) v1.0
- **Samples:** 7,043 customers
- **Features:** 20 (demographic, account information, and service details)
- **Target:** Binary classification (Churn: Yes/No)
- **Class Distribution:** Approximately 26% churn rate

## Model Performance

Performance metrics on held-out test set (20% of data):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9337 |
| Accuracy | 85.46% |
| Precision | 0.8536 |
| Recall | 0.8560 |
| F1-Score | 0.8548 |

**Best Model:** LightGBM  
**Training Time:** 0.84 minutes  
**Inference Latency:** <100ms per prediction

## Technical Stack

- **Python Libraries:** pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap, imbalanced-learn
- **Database:** SQLite (model registry and experiment tracking)
- **UI Framework:** Gradio
- **Visualization:** matplotlib, seaborn, plotly
- **Development:** Google Colab Pro with A100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload `Automated MLOps Framework for Customer Churn Prediction.ipynb` to Google Colab
3. Select Runtime > Change runtime type > A100 GPU (or T4 GPU for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Download and preprocess the dataset
- Train multiple models with hyperparameter optimization
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/Automated_MLOps_Framework_for_Customer_Churn_Prediction.git
cd Automated_MLOps_Framework_for_Customer_Churn_Prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm optuna shap imbalanced-learn gradio plotly seaborn matplotlib scipy joblib

# Run the notebook
jupyter notebook "Automated MLOps Framework for Customer Churn Prediction.ipynb"
```

## Project Structure

```
├── Automated MLOps Framework for Customer Churn Prediction.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

The notebook contains the following components:

1. **Configuration & Setup**: System configuration, logging, and reproducibility settings
2. **Database Management**: Model registry and experiment tracking
3. **Data Processing**: Loading, cleaning, and feature engineering
4. **Model Training**: Automated training pipeline with Optuna optimization
5. **Drift Detection**: Statistical tests for data distribution changes
6. **Evaluation**: Comprehensive performance metrics and visualizations
7. **Gradio Interface**: Interactive web application for predictions

## Key Implementation Details

- **Reproducibility:** All random seeds set to 42 for deterministic results
- **Cross-Validation:** 5-fold stratified cross-validation for model selection
- **Feature Engineering:** Automated creation of tenure groups, charge ratios, and service counts
- **Missing Data:** Median imputation for numerical features
- **Class Imbalance:** SMOTE oversampling applied to training data

## Limitations

- Trained specifically on telecommunications customer data; may not generalize to other industries
- Performance degrades with significant data drift (p-value < 0.05)
- Requires minimum 1,000 samples for reliable predictions
- Binary classification only (churn vs. no churn)
- Model performance may degrade over time without retraining

## Model Registry

The system maintains a SQLite database tracking:
- Model versions and hyperparameters
- Performance metrics on validation and test sets
- Training time and sample counts
- Production deployment status
- Drift detection results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IBM Telco Customer Churn dataset (Database Contents License v1.0)
- Kaggle community for dataset hosting and documentation
- Open-source libraries and frameworks used in this project

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate machine learning engineering and MLOps capabilities. Performance metrics are based on the specific dataset used and should be validated for any real-world application.*
