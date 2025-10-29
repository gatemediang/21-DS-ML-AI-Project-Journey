# Project 5: Customer Segmentation with Clustering 🛍️

## Problem Statement

A mall is seeking to gain a deeper understanding of its diverse customer base to implement more effective and personalized marketing strategies. The primary challenge is to identify distinct, homogeneous groups of customers based on their demographic attributes (Age, Gender) and spending behaviors (Annual Income, Spending Score). By segmenting customers, the mall aims to move away from a one-size-fits-all approach and instead tailor promotions, product offerings, store experiences, and communication channels to meet the specific needs and preferences of each identified segment, ultimately driving customer engagement, loyalty, and revenue.

## Solution Offered

This project addresses the problem of customer segmentation by applying **unsupervised machine learning clustering techniques** to the mall's customer dataset. The implemented solution involves a structured approach:

1.  **Data Loading and Initial Inspection:** The customer data is loaded into a pandas DataFrame, and initial checks are performed to understand its structure and identify any immediate issues (e.g., missing values, data types). The 'CustomerID' is dropped as it is not relevant for clustering.

2.  **In-Depth Exploratory Data Analysis (EDA):** Before clustering, extensive EDA is conducted to understand the distributions of individual features (univariate analysis) and the relationships between pairs of features (bivariate and multivariate analysis). Visualizations are heavily used in this phase to identify features that show clear patterns or groupings that could be good candidates for clustering.

3.  **K-Means Clustering:** The core clustering is performed using the K-Means algorithm.
    *   **Feature Selection and Scaling:** Relevant numerical features are selected and scaled using `StandardScaler` to ensure all features contribute equally to the distance calculations, which is crucial for K-Means.
    *   **Determining Optimal Clusters (Elbow Method):** The Elbow Method is implemented by calculating the Within-Cluster Sum of Squares (WCSS) for a range of cluster numbers (`k`). The WCSS values are plotted, and the "elbow point" is identified as the optimal number of clusters where the rate of decrease in WCSS significantly slows down.
    *   **Model Building and Fitting:** K-Means models are built and fitted to the scaled data using the determined optimal number of clusters. The `init='k-means++'` method is used for smart centroid initialization to improve convergence and avoid suboptimal solutions.
    *   **Cluster Assignment:** The trained K-Means model assigns each customer to a specific cluster, adding a new 'Cluster' column to the DataFrame.

4.  **Feature Engineering:** A new feature, 'Spending_Income_Ratio', is engineered by dividing the 'Spending Score' by the 'Annual Income'. This feature aims to capture spending behavior relative to income, potentially revealing different types of customers (e.g., those who spend a high proportion of their income vs. those who spend a low proportion). A small constant is added to the denominator to avoid division by zero.

5.  **Clustering with Engineered Feature:** K-Means clustering is also applied using the newly engineered 'Spending_Income_Ratio' feature in combination with 'Age' to explore alternative segmentation possibilities and gain different perspectives on the customer base. The Elbow Method is again used to determine the optimal number of clusters for this feature set.

6.  **Hierarchical Clustering (Alternative and Validation):** Hierarchical Clustering is introduced as an alternative clustering technique that does not require pre-specifying the number of clusters. A dendrogram is generated from the hierarchical clustering results. The dendrogram is used to visually confirm or validate the optimal number of clusters suggested by the Elbow Method for the K-Means models.

7.  **Persona Analysis and Interpretation:** For each clustering model, the characteristics of the identified clusters are analyzed quantitatively by calculating the mean values of the features (and the engineered feature) for each cluster. These quantitative profiles are then used to qualitatively interpret and define **Data-Driven Personas** for each segment, providing actionable insights for marketing strategies.

8.  **Summary of Findings:** A summary of the key insights derived from the entire analysis is provided, including findings from the initial EDA (like the most promising features for clustering), the characteristics of the different customer segments identified by the K-Means models (both with original and engineered features), and the analysis of Gender vs. Spending Score.

## Libraries Used and Why

*   **pandas:** Essential for efficient data loading, manipulation, cleaning, and preparation. It provides DataFrames, which are the primary data structure used.
*   **numpy:** Used for numerical computations, particularly in the feature engineering step to handle potential division by zero and for general array operations.
*   **matplotlib.pyplot:** A fundamental plotting library used for creating static visualizations such as the Elbow Method plot, box plots, and assisting in the creation of scatter plots. It provides fine-grained control over plot elements.
*   **seaborn:** Built on top of Matplotlib, Seaborn provides a high-level interface for drawing attractive statistical graphics. It was used for creating histograms, pair plots, box plots, and scatter plots with cluster assignments, making the visualizations more aesthetically pleasing and easier to generate for exploratory analysis and result presentation.
*   **plotly.express:** A high-level API for Plotly, used specifically for generating the interactive 3D scatter plot. This was crucial for visualizing the relationships between three numerical features simultaneously, allowing for interactive exploration of the data's structure in three dimensions.
*   **scipy.cluster.hierarchy:** A module within SciPy used for hierarchical clustering. It was specifically used to compute and plot the dendrogram, providing an alternative perspective on the data's cluster structure and aiding in validating the choice of `k`.
*   **sklearn.cluster.KMeans:** The primary class from scikit-learn used to perform K-Means clustering. It provides a robust and efficient implementation of the algorithm, including the `k-means++` initialization method.
*   **sklearn.preprocessing.StandardScaler:** Used to standardize features by removing the mean and scaling to unit variance. This preprocessing step is vital for distance-based algorithms like K-Means, as it prevents features with larger scales from disproportionately influencing the clustering results.

## Visualizations Used and Why

| Visualization Type             | Library Used          | Purpose                                                                                                                               |
| :----------------------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Histograms**                 | `seaborn.histplot`    | To visualize the distribution of individual numerical features (Age, Annual Income, Spending Score) and understand their shape and spread. Used with `hue='Gender'` to see gender-based differences. |
| **Pair Plot**                  | `seaborn.pairplot`    | To visualize the pairwise relationships between all numerical features. Crucial for identifying features with visible clusters (Income vs. Spending Score). Used with `hue='Gender'` to see gender influence on relationships. |
| **3D Scatter Plot**            | `plotly.express`      | To visualize the relationships between three key features (Annual Income, Spending Score, Age) in an interactive 3D space, providing a more comprehensive view of potential cluster structures. |
| **Elbow Method Plot**          | `matplotlib.pyplot`   | To plot WCSS vs. number of clusters (`k`) to visually identify the "elbow point" and determine the optimal number of clusters for K-Means. |
| **Scatter Plots (Clustered)**  | `seaborn.scatterplot` | To visualize the results of the K-Means clustering models by plotting data points colored by their assigned cluster. Essential for interpreting the distinct groups found by the algorithm. |
| **Box Plot**                   | `seaborn.boxplot`     | To compare the distribution of 'Spending Score' between different 'Gender' groups, quickly showing median, quartiles, and spread to assess gender impact on spending. |
| **Dendrogram**                 | `scipy.cluster.hierarchy` | To visualize the hierarchy of clusters built by Hierarchical Clustering. Helps in understanding cluster merging/splitting and can be used to validate the number of clusters. |

## Summary of Insights

The project yielded several key insights into the mall's customer base:

*   **Income vs. Spending is Key for Segmentation:** The pair plot and 3D visualization clearly showed distinct clusters forming based on 'Annual Income' and 'Spending Score'. This relationship proved to be the most effective for identifying well-separated customer segments.
*   **Optimal Clusters Vary by Feature Set:** The Elbow Method indicated that 5 clusters were optimal for segmentation based on 'Annual Income' and 'Spending Score', while 4 clusters were more appropriate for 'Age' and 'Spending Score', and also for 'Age' and the engineered 'Spending_Income_Ratio'. This highlights that the best way to segment customers depends on the dimensions being considered.
*   **Identified Customer Personas:** Based on the clustering results, several distinct customer segments were identified with actionable profiles, such as:
    *   High-Income, High-Spending (Target Customers)
    *   Low-Income, High-Spending (Enthusiasts)
    *   High-Income, Low-Spending (Careful High-Earners)
    *   Low-Income, Low-Spending (Budget Shoppers)
    *   Customers segmented by age groups and spending levels/ratios (e.g., Young High-Spenders, Older Low-Spenders, Younger customers with high spending relative to income, Older customers with lower spending relative to income).
*   **Gender has Limited Impact on Spending Score:** The analysis of 'Gender' vs. 'Spending Score' showed similar median spending scores for males and females, suggesting that gender alone is not a primary driver of spending habits in this dataset, although there were slight differences in variability and maximum spending.
*   **Hierarchical Clustering Validates K-Means:** The dendrogram from Hierarchical Clustering visually supported the choice of 5 clusters for the Income-Spending data, reinforcing the findings from the Elbow Method.
*   **Engineered Feature Provides New Perspective:** The 'Spending_Income_Ratio' feature, when used with 'Age', revealed different customer groupings compared to using original features, demonstrating the value of feature engineering in uncovering alternative segmentation dimensions.

These insights provide the mall with a data-driven understanding of its customer segments, enabling the marketing team to create more focused and effective campaigns.

## How to Use This Notebook

This notebook serves as a step-by-step guide to performing customer segmentation using clustering techniques.

1.  **Run All Cells:** Execute all code cells sequentially to reproduce the analysis.
2.  **Explore the Code:** Examine each code cell to understand the steps involved in data loading, EDA, feature scaling, clustering, and visualization.
3.  **Interpret Visualizations:** Study the generated plots (histograms, pair plots, 3D plot, elbow curves, scatter plots with clusters, box plot, dendrogram) to visually grasp the data distributions, relationships, and clustering results.
4.  **Analyze Cluster Profiles:** Review the quantitative cluster profiles (mean values of features per cluster) to understand the characteristics of each identified customer segment.
5.  **Read Markdown Cells:** Read the markdown cells for explanations of theoretical concepts (unsupervised learning, K-Means, Elbow Method, Hierarchical Clustering), step-by-step guidance through the analysis, and interpretation of results and personas.
6.  **Modify and Experiment:** Feel free to modify the code to:
    *   Try different numbers of clusters (`k`) in the K-Means models.
    *   Experiment with clustering on different combinations of features (both original and engineered).
    *   Explore other clustering algorithms available in scikit-learn.
    *   Create different engineered features.
    *   Refine the visualization parameters.
7.  **Apply to Your Own Data:** Use this notebook as a template to apply customer segmentation (or other clustering tasks) to your own datasets, adapting the feature selection, scaling, and analysis steps as needed.

## License
* MIT