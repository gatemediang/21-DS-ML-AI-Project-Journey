# End-to-End Exploratory Data Analysis (EDA) on the Titanic Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FNq5-xSNv_CDABhmkk1ki995T2MOpZL1?authuser=1#scrollTo=362d1908)

## 1. Problem Statement

The sinking of the RMS Titanic is one of the most infamous maritime disasters in history. This project aims to explore and analyze the dataset containing information about the passengers aboard the Titanic to identify key factors that influenced survival. The core problem is to understand *who* survived and *why*, using data analysis techniques to uncover patterns and relationships within the data.

## 2. Solution Offered

This notebook provides an end-to-end exploratory data analysis (EDA) of the Titanic dataset. It covers the entire process from initial data loading and inspection to detailed univariate, bivariate, and multivariate analysis, including feature engineering and correlation analysis. The goal is to provide a comprehensive understanding of the dataset and the factors related to survival, serving as a foundational step for potential future modeling efforts.
Google Colab: [Open in Colab](https://colab.research.google.com/drive/1FNq5-xSNv_CDABhmkk1ki995T2MOpZL1?authuser=1#scrollTo=362d1908)
Titanic Dataset Profiling Report: [data_report.html](data_report.html)

## 3. Step-by-Step Analysis

The analysis in this notebook follows a structured approach:

### Step 1: Setup - Importing Libraries

Essential Python libraries for data manipulation (`pandas`, `numpy`) and visualization (`matplotlib`, `seaborn`) are imported to begin the analysis.

### Step 2: Data Loading and Initial Inspection

The Titanic dataset (`Titanic-Dataset.csv`) is loaded into a pandas DataFrame. Initial inspection using `.head()`, `.tail()`, `.shape`, and `.info()` reveals the dataset's structure, the presence of missing values (in 'Age', 'Cabin', and 'Embarked'), and basic statistics. `.describe()` provides insights into the distribution of numerical features, highlighting the low survival rate (~38.4%) and the skewed distribution of 'Fare'.

**Insights:**
- The dataset contains 891 passengers and 12 features.
- Significant missing data exists, particularly in the 'Cabin' column.
- The majority of passengers did not survive.
- Fare distribution is heavily skewed, indicating outliers.

### Step 3: Data Cleaning

Missing values in 'Age' and 'Embarked' are handled. 'Age' is imputed with the median (28.0), and 'Embarked' is filled with the mode ('S'). The 'Cabin' column, with a high percentage of missing data, is transformed into a new binary feature 'Has_Cabin' (1 if cabin information is available, 0 otherwise), and the original 'Cabin' column is dropped.

**Insights:**
- Missing 'Age' values are addressed by median imputation.
- Missing 'Embarked' values are addressed by mode imputation.
- The 'Cabin' feature's missingness is handled by creating a binary indicator, preserving some information without complex imputation.

### Step 4: Univariate Analysis

Each feature is analyzed individually using count plots and histograms to understand their distributions. This step visualizes the counts of categories for categorical features and the distribution of values for numerical features.

**Key Insights:**
- **Categorical:** Most passengers were in 3rd class, male, and embarked from Southampton. Most traveled alone or in small groups.
- **Numerical:** Age distribution peaks around 20-30. Fare distribution is heavily right-skewed.

### Step 5: Bivariate Analysis

The relationship between pairs of variables is explored, focusing on how each feature relates to the target variable, 'Survived'. Bar plots and violin plots are used for visualization.

**Key Insights:**
- **Pclass:** Higher class correlated with higher survival rate.
- **Sex:** Females had a significantly higher survival rate than males across all classes.
- **Embarked:** Passengers from Cherbourg ('C') had a higher survival rate.
- **Has_Cabin:** Having a cabin was associated with a higher survival rate.
- **Age vs. Survival:** Young children (especially males) and women of most ages had higher survival rates.

### Step 6: Feature Engineering

New features are created from existing ones:
- **FamilySize:** Combining 'SibSp' and 'Parch' plus one (for the passenger themselves).
- **IsAlone:** A binary indicator based on 'FamilySize' (1 if FamilySize is 1, 0 otherwise).
- **Title:** Extracted from the 'Name' column and simplified by grouping rare titles.

**Insights:**
- **FamilySize & IsAlone:** Traveling in small families (2-4 members) had the highest survival rates, while those alone or in very large families had lower rates.
- **Title:** 'Mrs' and 'Miss' had high survival rates, while 'Mr' had a low rate. 'Master' had a higher survival rate than 'Mr'.

### Step 7: Multivariate Analysis

Interactions between multiple variables are examined. Examples include visualizing survival rate by Pclass and Sex, and Age distribution by Sex and Survival status.

**Insights:**
- The 'women and children first' protocol is clearly visible across different classes.
- The age distribution of survivors vs. non-survivors differs significantly between sexes.

### Step 8: Correlation Analysis

A correlation heatmap is generated for numerical features to visualize the linear relationships between them and with the 'Survived' target variable.

**Interpretation:**
- 'Survived' is positively correlated with 'Fare' and 'Has_Cabin', and negatively correlated with 'Pclass' and 'IsAlone'.
- Strong negative correlation between 'Pclass' and 'Fare'.

### Step 9: Generating Profiling Report

The `ydata-profiling` library is used to generate a comprehensive interactive HTML report summarizing the dataset. This report provides detailed statistics, visualizations, and insights for each variable and their interactions. The report is displayed within the notebook and can also be exported to an HTML file (`data_report.html`).

## Key Findings from EDA

- Survival Rate: Only about 38.4% of the passengers in this dataset survived the Titanic disaster.
- Passenger Class (Pclass): Passenger class was a strong indicator of survival. First-class passengers had a significantly higher survival rate compared to those in second and third class.
- Sex: Gender was the most influential factor in survival. Females had a much higher survival rate (around 75%) than males (below 20%). This highlights the "women and children first" protocol.
- Age: Age played a role in survival. Infants and young children had a better chance of surviving, while a large proportion of non-survivors were young adults.
- Embarked: Passengers who embarked from Cherbourg ('C') had a higher survival rate than those from Southampton ('S') or Queenstown ('Q').
- Has_Cabin: Passengers for whom cabin information was available had a higher survival rate. This is likely correlated with being in a higher passenger class.
- Family Size and Alone Status: Traveling in small families (2–4 people) was associated with the highest survival rates. Passengers traveling alone or in very large families had lower survival rates.
- Title: The title extracted from names provided valuable insight. 'Mrs' and 'Miss' (mostly females) had high survival rates, while 'Mr' (mostly males) had a low survival rate. 'Master' (young boys) also had a relatively higher survival rate than adult males.
- Fare: There was a positive correlation between fare and survival, indicating that passengers who paid higher fares (likely in higher classes) were more likely to survive. The fare distribution was heavily skewed, with many low fares and a few very high fares.
- Missing Data: Significant missing data was found in the 'Age', 'Cabin', and 'Embarked' columns. Age and Embarked were imputed, while 'Cabin' was transformed into a 'Has_Cabin' feature due to the high percentage of missing values.

These findings suggest that socio-economic status (Pclass, Fare, Has_Cabin) and demographic factors (Sex, Age, Title, Family Size) were significant factors influencing survival on the Titanic.

## How to Run the Notebook

To run this notebook and reproduce the analysis:

1.  **Open in Google Colab:** Click on the "Open in Colab" badge if available or upload the `.ipynb` file to your Google Drive and open it with Google Colaboratory.
2.  **Run Cells:** Execute each code cell sequentially by clicking the "play" button on the left of the cell or by pressing `Shift + Enter`.
3.  **Data Loading:** The notebook includes code to clone the dataset from a GitHub repository. Ensure this cell is executed to download the data.
4.  **Dependencies:** The notebook automatically installs the `ydata-profiling` library. All other required libraries (`pandas`, `numpy`, `seaborn`, `matplotlib`) are pre-installed in Colab.
5.  **View Output:** The outputs, including visualizations and the profiling report, will be displayed directly within the notebook cells after execution. The HTML profiling report will also be saved to your Colab environment, which you can download.

## License

MIT