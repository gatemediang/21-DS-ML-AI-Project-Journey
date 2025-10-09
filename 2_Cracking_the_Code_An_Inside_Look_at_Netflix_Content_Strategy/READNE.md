# Project 2: In-Depth Exploratory Data Analysis (EDA) - Netflix Content Analysis 🎬

## Project Objective

To perform an in-depth exploratory data analysis of the Netflix dataset. We will explore trends in content production, identify popular genres, analyze content ratings, and understand the distribution of movies and TV shows on the platform. This project builds on foundational EDA by introducing time-series analysis and more complex data cleaning and transformation techniques.

## Problem Statement

The goal of this project is to understand the composition and evolution of the Netflix content library. By analyzing the provided dataset, we aim to uncover trends in content acquisition, identify the types of content Netflix prioritizes, understand the geographical distribution of content production, and gain insights into the target audience based on content ratings.

## Solution Offered

This exploratory data analysis utilizes Python and several powerful libraries to clean, transform, analyze, and visualize the Netflix content data.

**Libraries Used:**

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
*   `seaborn`: For creating informative statistical graphics.
*   `wordcloud`: For generating word cloud visualizations.
*   `nltk`: For natural language processing tasks like tokenization and stop word removal.

**Tools Used:**

*   Google Colab: The environment where the analysis was performed.

**Charts Used:**

*   Pie Chart (Proportion of Movies vs. TV Shows)
*   Line Plot (Content Added to Netflix Over the Years)
*   Bar Plot (Top Genres)
*   Histograms (Movie Duration, Content Age)
*   Box Plot (Content Age by Type)
*   Hexbin Plot (Release Year vs. Year Added)
*   Bar Plot (Top Directors)
*   Word Cloud (Most Common Words in Descriptions)
*   Area Plot (Distribution of Content Ratings Over Time)

[Open this notebook in google colab](https://colab.research.google.com/drive/1ymNGPdKERT_XismWba_QTAJaDi-Q816V?authuser=1#scrollTo=lT19alljTPRa)

## Exploratory Data Analysis Steps

The analysis was carried out through the following steps:

1.  **Setup - Importing Libraries:** Importing necessary libraries for data manipulation, visualization, and text analysis.
2.  **Data Loading and Initial Inspection:** Loading the `netflix_titles.csv` dataset and performing initial checks on its structure, data types, and missing values using `.head()`, `.shape()`, and `.info()`.
3.  **Data Cleaning and Transformation:** Handling missing values in columns like 'director', 'cast', 'country', 'date_added', and 'rating'. This involved filling missing values with 'Unknown' or the mode, and dropping rows with missing essential information. The `date_added` column was converted to a datetime object for time-series analysis.
4.  **Exploratory Data Analysis & Visualization:** Analyzing and visualizing key aspects of the dataset, including the distribution of content types, content added over time, popular genres, and content duration.
5.  **Feature Engineering - Content Freshness:** Creating a new feature `age_on_netflix` to represent the difference between the year added to Netflix and the release year, providing insights into content acquisition strategy.
6.  **Deeper Multivariate Analysis:** Examining relationships between variables, such as movie duration across different genres.
7.  **Word Cloud from Content Descriptions:** Generating a word cloud to visualize the most frequent terms used in content descriptions, revealing common themes.
8.  **Answering Submission Questions:** Addressing specific questions about the data, including changes in content ratings over time, the relationship between content age and type, trends in production based on release year vs. year added, common word pairs in descriptions, and top directors.

## Key Findings and Insights

Based on the comprehensive EDA, the following key insights were revealed:

*   **Content Mix:** Movies dominate the Netflix library (approx. 70%), with a significant increase in content additions observed from 2016 to 2019. Netflix employs a mixed strategy of adding both recent releases ("Netflix Originals") and older licensed content.
*   **Global Reach:** While the United States is the largest content producer, India stands out as a major contributor, emphasizing Netflix's global content strategy. Other significant contributors include the UK, Japan, and South Korea.
*   **Audience Focus:** The content library is primarily geared towards mature audiences, with `TV-MA` and `TV-14` being the most prevalent ratings. However, there is also a notable increase in kids' content (`TV-Y`, `TV-Y7`).
*   **Genre Popularity and Format:** "International Movies," "Dramas," and "Comedies" are the most popular genres. Most movies have a standard duration of 80-120 minutes, while the majority of TV shows are short, with most having only one season.
*   **Common Themes:** Content descriptions frequently feature themes related to human relationships, personal journeys, urban settings, friendships, and stories based on real events.
*   **Top Directors:** Directors specializing in stand-up comedy specials have a strong presence among the top directors by title count, highlighting Netflix's investment in this genre. Prominent feature film directors and international directors are also well-represented.

## Questions Answered

Here are the specific questions addressed during the EDA and their summaries:

*   **How has the distribution of content ratings changed over time?**
    The distribution shows a significant increase in titles across most ratings from 2016 onwards. `TV-MA` and `TV-14` dominate later years, indicating a focus on mature content, while kids' ratings (`TV-Y`, `TV-Y7`) also increased but remain lower in volume. Older ratings appear relatively stable or slightly increasing.
*   **Is there a relationship between content age and its type (Movie vs. TV Show)?**
    Movies have a wider distribution of ages when added to Netflix, including many older films. TV shows have a much tighter distribution, with most being added soon after release, suggesting a focus on newer content for TV series.
*   **Can we identify any trends in content production based on the release year vs. the year added to Netflix?**
    A strong diagonal trend shows a large amount of content is added in the same year or shortly after release, especially in recent years (from 2015 onwards), reflecting Netflix's original content strategy. The plot also shows the addition of older library content across various release years.
*   **What are the most common word pairs or phrases in content descriptions?**
    Common word pairs include "high school," "young man," "young woman," "new york," "best friends," and "true story," highlighting common themes of coming-of-age, urban settings, relationships, and real events.
*   **Who are the top directors on Netflix?**
    The top directors include those specializing in stand-up comedy specials as well as prominent feature film and international directors, reflecting the diversity of content and genres on the platform.

## How to Reproduce the Analysis

To reproduce this analysis, you can follow these steps:

1.  Open the provided Google Colab notebook using the link above.
2.  Run each code cell sequentially. The notebook contains all the necessary code for data loading, cleaning, transformation, analysis, and visualization.
3.  Ensure you have an active internet connection to clone the dataset from the GitHub repository.
4.  Review the markdown cells for explanations and insights at each step of the analysis.

This notebook is designed to be self-contained and executable in the Google Colab environment.