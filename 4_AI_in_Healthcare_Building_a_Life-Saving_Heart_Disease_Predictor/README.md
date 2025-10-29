# Heart Disease Prediction with Machine Learning 🩺

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF)](https://www.kaggle.com/)

## 📋 Overview

A comprehensive machine learning classification system for predicting heart disease in patients based on medical attributes. This project demonstrates the complete ML pipeline from exploratory data analysis to model deployment, comparing multiple classification algorithms to achieve accurate and reliable predictions for early diagnosis and intervention.

## 🎯 Problem Statement

### The Challenge

Build a machine learning model that can accurately predict whether a patient has heart disease based on a set of medical attributes. This is a **binary classification problem** where the model needs to distinguish between:
- **Positive Class (1)**: Patient has heart disease
- **Negative Class (0)**: Patient does not have heart disease

### Medical Context

Early detection of heart disease is critical for:
- **Prevention**: Identifying high-risk patients before complications occur
- **Treatment Planning**: Enabling timely medical interventions
- **Resource Allocation**: Prioritizing patients who need immediate care
- **Cost Reduction**: Preventing expensive emergency treatments through early diagnosis

### Dataset Details

- **Source**: Kaggle Heart Disease Dataset
- **Features**: 13 medical attributes (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target Variable**: Binary indicator of heart disease presence
- **Total Observations**: [Number from dataset]
- **Feature Types**: Mix of categorical and numerical features

## 💡 Solution Approach

Our solution implements a comprehensive machine learning pipeline with multiple classification algorithms:

### 1. **Data Loading and Inspection**
- Load dataset using `kagglehub`
- Perform initial structure analysis
- Check data types and identify missing values
- Validate data integrity

### 2. **Exploratory Data Analysis (EDA)**
- **Target Distribution Analysis**: Understanding class balance
- **Feature Distribution**: Analyzing categorical and numerical features
- **Correlation Analysis**: Identifying relationships between features
- **Visualization**: Creating informative plots for insights

### 3. **Data Preprocessing**

**Two Approaches Demonstrated**:

**A. Pipeline-Based Approach** (Recommended)
- Integrated preprocessing and modeling
- Prevents data leakage
- Ensures consistent transformations

**B. Manual Preprocessing**
- Explicit control over each step
- Better for understanding the process

**Preprocessing Steps**:
- Missing value imputation using `SimpleImputer`
- One-hot encoding for categorical features
- Standard scaling for numerical features

### 4. **Model Building and Training**

**Algorithms Implemented**:
1. **Logistic Regression** - Baseline linear model
2. **Random Forest Classifier** - Ensemble tree-based model
3. **Support Vector Machine (SVM)** - Kernel-based classification
4. **K-Nearest Neighbors (KNN)** - Instance-based learning

### 5. **Model Evaluation**

**Metrics Used**:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### 6. **Feature Importance Analysis**

Using Random Forest to identify the most influential medical attributes for prediction.

## 🔧 Technology Stack

### Core Libraries

#### Data Manipulation & Analysis
- **pandas** (`pd`): Data loading, manipulation, and preprocessing
  - *Why*: Efficient DataFrame operations for handling medical datasets
  - *Usage*: Loading CSV, handling missing values, feature engineering

- **numpy** (`np`): Numerical operations and array manipulations
  - *Why*: Fast mathematical computations and array operations
  - *Usage*: Matrix operations, statistical calculations

#### Visualization
- **matplotlib.pyplot** (`plt`): Creating static visualizations
  - *Why*: Foundational plotting library for customizable charts
  - *Usage*: Distribution plots, confusion matrix visualization

- **seaborn** (`sns`): Statistical data visualization
  - *Why*: High-level interface for attractive statistical graphics
  - *Usage*: Heatmaps, count plots, correlation matrices, box plots

#### Dataset Access
- **kagglehub**: Direct dataset download from Kaggle
  - *Why*: Seamless integration with Kaggle datasets
  - *Usage*: Automated dataset downloading and caching

#### Machine Learning - Preprocessing

- **sklearn.model_selection.train_test_split**: Dataset splitting
  - *Why*: Creates training and testing sets with stratified sampling
  - *Usage*: 80/20 split for model training and evaluation

- **sklearn.preprocessing.StandardScaler**: Feature scaling
  - *Why*: Normalizes features to zero mean and unit variance
  - *Usage*: Scaling numerical features (age, blood pressure, cholesterol)

- **sklearn.preprocessing.OneHotEncoder**: Categorical encoding
  - *Why*: Converts categorical variables to numerical format
  - *Usage*: Encoding chest pain type, ECG results, thalassemia type

- **sklearn.compose.ColumnTransformer**: Selective preprocessing
  - *Why*: Applies different transformers to different columns
  - *Usage*: Simultaneous scaling and encoding pipeline

- **sklearn.pipeline.Pipeline**: Workflow chaining
  - *Why*: Chains preprocessing steps and model into single object
  - *Usage*: Prevents data leakage, simplifies deployment

- **sklearn.impute.SimpleImputer**: Missing value handling
  - *Why*: Fills missing values using simple strategies (mean/median/mode)
  - *Usage*: Imputing missing medical measurements

#### Machine Learning - Models

- **sklearn.linear_model.LogisticRegression**: Baseline classifier
  - *Why*: Simple, interpretable, fast training
  - *Usage*: Establishing performance baseline

- **sklearn.ensemble.RandomForestClassifier**: Ensemble model
  - *Why*: Handles non-linearity, provides feature importance
  - *Usage*: Primary model for high-accuracy predictions

- **sklearn.svm.SVC**: Support Vector Classifier
  - *Why*: Effective in high-dimensional spaces
  - *Usage*: Kernel-based classification for complex boundaries

- **sklearn.neighbors.KNeighborsClassifier**: Instance-based learner
  - *Why*: Simple, non-parametric approach
  - *Usage*: Comparison model for distance-based classification

#### Machine Learning - Evaluation

- **sklearn.metrics.accuracy_score**: Overall accuracy calculation
  - *Why*: Measures proportion of correct predictions
  
- **sklearn.metrics.confusion_matrix**: Error analysis
  - *Why*: Visualizes true/false positives and negatives
  
- **sklearn.metrics.classification_report**: Comprehensive metrics
  - *Why*: Displays precision, recall, F1-score per class
  
- **sklearn.metrics.precision_score**: Precision calculation
  - *Why*: Measures positive prediction reliability
  
- **sklearn.metrics.recall_score**: Recall/Sensitivity calculation
  - *Why*: Measures ability to find all positive cases
  
- **sklearn.metrics.f1_score**: F1-Score calculation
  - *Why*: Balances precision and recall

## 🚀 Local Implementation Guide

### Prerequisites

- Python 3.8 or higher
- Kaggle account (for dataset access)
- 4GB+ RAM recommended

### Step 1: Environment Setup

**Option A: Using Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv heart_disease_env

# Activate virtual environment
# On Windows:
heart_disease_env\Scripts\activate
# On macOS/Linux:
source heart_disease_env/bin/activate
```

**Option B: Using Conda**

```bash
# Create conda environment
conda create -n heart_disease python=3.8

# Activate environment
conda activate heart_disease
```

### Step 2: Install Required Libraries

```bash
# Install all dependencies
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter

# Verify installations
python -c "import pandas, numpy, sklearn, kagglehub; print('All packages installed successfully!')"
```

**Dependencies List** ([`requirements.txt`](requirements.txt)):
```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

Install from file:
```bash
pip install -r requirements.txt
```

### Step 3: Kaggle API Setup (Optional)

If using Kaggle API for dataset download:

1. **Get API Credentials**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to "API" section
   - Click "Create New Token"
   - Download `kaggle.json`

2. **Configure Kaggle CLI**:

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle  # macOS/Linux
# or
mkdir %USERPROFILE%\.kaggle  # Windows

# Move kaggle.json to the directory
# macOS/Linux:
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### Step 4: Download Dataset

The notebook uses `kagglehub` to automatically download the dataset:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("dataset-name")
print("Path to dataset files:", path)
```

**Manual Download Alternative**:
1. Visit the Kaggle dataset page
2. Download the CSV file
3. Place it in the project directory

### Step 5: Run the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open: 4_AI_in_Healthcare_Building_a_Life-Saving_Heart_Disease_Predictor.ipynb
# Execute cells sequentially (Shift+Enter)
```

### Step 6: Review Results

After running all cells, you'll see:
- Model performance metrics for all algorithms
- Confusion matrices
- Feature importance rankings
- Visualizations of data distributions

## 📊 Project Structure

```
4_AI_in_Healthcare_Building_a_Life-Saving_Heart_Disease_Predictor/
├── 4_AI_in_Healthcare_Building_a_Life-Saving_Heart_Disease_Predictor.ipynb
├── data/
│   └── heart_disease.csv         # Dataset (auto-downloaded)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔍 Key Insights & Findings

### 1. Target Variable Distribution

**Class Balance Analysis**:
- Distribution of patients with and without heart disease
- Implications for model training and evaluation
- Need for balanced accuracy metrics

### 2. Categorical Feature Insights

#### **Sex Distribution**
- **Finding**: Dataset contains significantly more male patients than female patients
- **Implication**: Gender may be a significant predictor; model should be validated across genders

#### **Chest Pain Type (cp)**
- **Most Common**: Asymptomatic chest pain
- **Distribution**: Asymptomatic > Non-anginal pain > Atypical angina > Typical angina
- **Clinical Significance**: Different pain types indicate varying disease severity

#### **Fasting Blood Sugar (fbs)**
- **Finding**: Most patients have fasting blood sugar ≤ 120 mg/dl
- **Implication**: Diabetes (indicated by high fasting blood sugar) is less prevalent in this dataset

#### **Resting ECG Results (restecg)**
- **Distribution**: Normal > Left ventricular hypertrophy > ST-T abnormality
- **Insight**: Most patients have normal resting ECG results

#### **Exercise Induced Angina (exang)**
- **Finding**: More patients without exercise-induced angina
- **Clinical Note**: Exercise-induced angina is a strong heart disease indicator

#### **ST Segment Slope (slope)**
- **Distribution**: Flat > Upsloping > Downsloping
- **Medical Context**: Flat and downsloping ST segments often indicate ischemia

#### **Thalassemia (thal)**
- **Distribution**: Normal > Reversible defect > Fixed defect
- **Rarity**: Fixed defects are least common but clinically significant

### 3. Numerical Feature Insights

#### **Age Distribution**
- **Pattern**: Somewhat normal distribution, peak around late 50s
- **Observation**: Heart disease risk increases with age

#### **Resting Blood Pressure (trestbps)**
- **Distribution**: Right-skewed, most values between 120-140 mmHg
- **Clinical Range**: Values within prehypertension to stage 1 hypertension range

#### **Serum Cholesterol (chol)**
- **Distribution**: Right-skewed with lower-end clustering
- **Data Quality Issue**: Zero values may indicate missing data or measurement errors

#### **Maximum Heart Rate (thalch)**
- **Distribution**: Relatively normal, centered around 140-150 bpm
- **Insight**: Exercise capacity is a key cardiac health indicator

#### **ST Depression (oldpeak)**
- **Distribution**: Heavily skewed towards lower values, peak at 0
- **Interpretation**: Most patients have minimal ST depression at rest

#### **Major Vessels (ca)**
- **Distribution**: Heavily skewed towards 0
- **Finding**: Most patients have 0 major vessels colored by fluoroscopy
- **Clinical Significance**: Number of vessels correlates with disease severity

### 4. Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Random Forest | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| SVM | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| KNN | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

*(Note: Fill in actual metrics after running the notebook)*

### 5. Feature Importance Analysis

**Top Predictive Features** (from Random Forest):
1. [Feature 1]: [Importance score]
2. [Feature 2]: [Importance score]
3. [Feature 3]: [Importance score]
4. [Feature 4]: [Importance score]
5. [Feature 5]: [Importance score]

*(Note: Update after running feature importance analysis)*

### 6. Clinical Implications

**Key Takeaways for Medical Practice**:
- **Most Important Features**: [List top features that doctors should monitor]
- **Risk Factors**: [Highlight modifiable risk factors]
- **Early Warning Signs**: [Features that indicate high risk]
- **Model Limitations**: [Discuss what the model cannot predict]

## 📈 Model Configuration

### Random Forest Hyperparameters

```python
RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=None,             # No depth limit
    min_samples_split=2,        # Minimum samples to split node
    min_samples_leaf=1,         # Minimum samples in leaf node
    random_state=42             # Reproducibility
)
```

### Logistic Regression Configuration

```python
LogisticRegression(
    max_iter=1000,              # Maximum iterations
    random_state=42,            # Reproducibility
    solver='lbfgs'              # Optimization algorithm
)
```

### SVM Configuration

```python
SVC(
    kernel='rbf',               # Radial basis function kernel
    C=1.0,                      # Regularization parameter
    random_state=42             # Reproducibility
)
```

### KNN Configuration

```python
KNeighborsClassifier(
    n_neighbors=5,              # Number of neighbors
    weights='uniform',          # Equal weights for all neighbors
    metric='euclidean'          # Distance metric
)
```

## 🎓 Learning Outcomes

### Concepts Mastered

1. **Binary Classification**: Understanding two-class prediction problems
2. **Medical Data Analysis**: Working with healthcare datasets
3. **Feature Engineering**: Creating and selecting relevant features
4. **Imbalanced Data Handling**: Techniques for skewed class distributions
5. **Model Comparison**: Evaluating multiple algorithms systematically
6. **Pipeline Construction**: Building reproducible ML workflows
7. **Feature Importance**: Interpreting model decisions

### Skills Developed

- ✅ End-to-end classification pipeline construction
- ✅ Medical domain knowledge application
- ✅ Multiple algorithm implementation and comparison
- ✅ Comprehensive model evaluation
- ✅ Data preprocessing and transformation
- ✅ Visualization of medical data
- ✅ Clinical insight extraction from ML models

## 🚧 Potential Improvements

### Next Steps for Better Performance

1. **Hyperparameter Tuning**:
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [None, 10, 20, 30],
       'min_samples_split': [2, 5, 10]
   }
   
   grid_search = GridSearchCV(
       RandomForestClassifier(random_state=42),
       param_grid,
       cv=5,
       scoring='f1'
   )
   ```

2. **Advanced Ensemble Methods**:
   - Implement XGBoost or LightGBM
   - Use voting classifiers
   - Stack multiple models

3. **Feature Engineering**:
   - Create interaction terms (e.g., age × cholesterol)
   - Polynomial features for key predictors
   - Domain-specific transformations

4. **Class Imbalance Handling**:
   ```python
   from imblearn.over_sampling import SMOTE
   
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

5. **Cross-Validation**:
   ```python
   from sklearn.model_selection import cross_val_score
   
   scores = cross_val_score(
       model,
       X_train,
       y_train,
       cv=5,
       scoring='f1'
   )
   print(f"CV F1-Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
   ```

6. **Explainable AI**:
   ```python
   import shap
   
   explainer = shap.TreeExplainer(rf_model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

## 📚 Additional Resources

### Medical Context
- [American Heart Association - Heart Disease Statistics](https://www.heart.org/en/health-topics/heart-attack)
- [WHO - Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
- [Understanding ECG Results](https://www.nhlbi.nih.gov/health/heart-tests/electrocardiogram)

### Machine Learning
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Feature Importance in Random Forests](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

### Model Interpretation
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Interpreting ML Models in Healthcare](https://www.nature.com/articles/s41746-019-0099-x)

## 🤝 Contributing

Contributions to improve the model or add new features are welcome!

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/ImprovedModel`)
3. Make your changes to the notebook
4. Test your modifications
5. Commit your changes (`git commit -m 'Add improved preprocessing'`)
6. Push to the branch (`git push origin feature/ImprovedModel`)
7. Open a Pull Request

### Contribution Ideas

- Implement additional classification algorithms (XGBoost, CatBoost)
- Add SHAP values for model interpretability
- Create interactive dashboards for EDA
- Develop API endpoint for model deployment
- Add unit tests for preprocessing functions
- Implement cross-validation with different strategies

## 🐛 Troubleshooting

### Common Issues

**Issue: "No module named 'kagglehub'"**
```bash
# Solution:
pip install kagglehub
```

**Issue: Dataset download fails**
```bash
# Solution: Download manually from Kaggle
# 1. Visit dataset page on Kaggle
# 2. Click "Download" button
# 3. Place CSV file in project directory
```

**Issue: Imbalanced classes affecting model**
```python
# Solution: Use stratified split and class weights
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintains class distribution
)

# Use class_weight parameter
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

**Issue: Zero values in cholesterol**
```python
# Solution: Treat zeros as missing values
df['chol'] = df['chol'].replace(0, np.nan)

# Then impute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df['chol'] = imputer.fit_transform(df[['chol']])
```

## 📄 License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025


## 🙏 Acknowledgments

- **Kaggle Community**: For providing the heart disease dataset
- **UCI Machine Learning Repository**: Original source of the dataset
- **Scikit-learn Team**: For excellent ML tools
- **Medical Professionals**: For domain expertise in feature interpretation
- **Open Source Community**: For continuous improvements to ML libraries

## 📞 Support

For questions or issues:

1. **GitHub Issues**: Open an issue in this repository
2. **Kaggle Discussion**: Post in dataset discussion section
3. **Documentation**: Check library documentation for API details
4. **Medical Questions**: Consult healthcare professionals for clinical advice

---

**Built with ❤️ for advancing healthcare through machine learning**

*⚠️ Disclaimer: This model is for educational purposes only and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.*