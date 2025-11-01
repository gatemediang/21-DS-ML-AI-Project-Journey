# Customer Churn Prediction with Feature Engineering and Selection

## Project Objective
To demonstrate the power of feature engineering by building and comparing two models: a baseline model with raw features and an enhanced model with newly engineered features. The goal is to accurately predict customer churn for a telecommunications company.

## Accomplished Tasks
- Data Cleaning and Initial Preparation
- Baseline Model Creation and Evaluation
- Feature Engineering
- Feature Selection Experimentation
- Alternative Model Evaluation
- Hyperparameter Tuning
- Performance Comparison and Analysis

## Solutions Implemented
- **Advanced Data Cleaning:** Handled missing values in `TotalCharges` by converting the column to numeric and imputing with the median. Converted the target variable `Churn` to a binary format.
- **Feature Engineering:** Created several new features including `tenure_group` (binned tenure), simplified categorical variables, `num_add_services` (count of additional services), `monthly_charge_ratio`, `total_to_monthly_charges_ratio`, and interaction terms like `has_internet_security`, `monthly_charge_long_term`, and `senior_fiber_optic`.
- **Model Building Pipeline:** Utilized `ColumnTransformer` and `Pipeline` from Scikit-Learn for robust preprocessing and model training.
- **Model Evaluation:** Evaluated multiple classification models (Logistic Regression, Random Forest, Gradient Boosting) using classification reports and accuracy scores, focusing on the F1-score for the churn class.
- **Feature Selection:** Explored feature selection using `SelectFromModel` (with different thresholds) and `SelectKBest` on the engineered feature set.
- **Hyperparameter Tuning:** Performed hyperparameter tuning on promising models (Logistic Regression, Gradient Boosting) using `GridSearchCV` to optimize performance.

## Libraries Used
- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `matplotlib.pyplot` for basic plotting.
- `seaborn` for enhanced data visualization.
- `sklearn` (Scikit-Learn) for:
    - `model_selection`: `train_test_split`, `GridSearchCV`
    - `preprocessing`: `StandardScaler`, `OneHotEncoder`
    - `compose`: `ColumnTransformer`
    - `pipeline`: `Pipeline`
    - `linear_model`: `LogisticRegression`
    - `ensemble`: `RandomForestClassifier`, `GradientBoostingClassifier`
    - `metrics`: `accuracy_score`, `classification_report`, `confusion_matrix`
    - `feature_selection`: `SelectFromModel`, `SelectKBest`, `f_classif`, `chi2`

## Visualizations
- **Feature Importance Plot:** A bar plot showing the top 15 most important features based on the Random Forest model trained on engineered features, indicating which features have the most influence on churn prediction.

## Key Insights and Findings
- Feature engineering significantly increased the number of features and generally improved model performance, particularly in terms of the F1-score for the churn class compared to the baseline.
- Feature selection methods explored did not consistently outperform using the full set of engineered features for the evaluated models.
- Logistic Regression and Gradient Boosting models performed well on the engineered dataset.
- Hyperparameter tuning provided marginal improvements for Logistic Regression but not for Gradient Boosting within the tested parameter ranges.
- The most influential features for churn prediction include contract type (month-to-month), internet service (fiber optic), tenure, monthly and total charges, lack of online security and tech support, payment method (electronic check), and the number of services.

## How to Use the Code
1.  **Clone the repository:** If the code is in a repository, clone it to your local machine. (In this notebook environment, the data was cloned from a specified URL).
2.  **Ensure dependencies are installed:** Make sure you have the necessary Python libraries installed (pandas, numpy, scikit-learn, matplotlib, seaborn). You can typically install them using pip: `pip install pandas numpy scikit-learn matplotlib seaborn`.
3.  **Run the Jupyter Notebook:** Open the `.ipynb` file in a Jupyter Notebook or JupyterLab environment.
4.  **Execute cells sequentially:** Run the code cells in order from top to bottom. The notebook is structured to follow the steps of data loading, cleaning, feature engineering, modeling, evaluation, and analysis.

## License
MIT
