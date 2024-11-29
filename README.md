# Loan Default Prediction
This repository showcases an end-to-end project focused on predicting loan defaults using a combination of Exploratory Data Analysis (EDA) and Machine Learning (ML) techniques. The project addresses a critical challenge faced by financial institutions: identifying high-risk applicants while ensuring capable customers are not rejected. The goal is to analyze patterns, gain insights, and implement predictive models to support informed decision-making in loan approvals.

* Project Description
Imagine you're a data analyst at a financial institution that specializes in lending various types of loans. The company faces two major risks:

  Losing Business: If capable applicants are rejected.
  Financial Loss: If risky applicants are approved and default on their loans.
  To mitigate these risks, we analyze a dataset of loan applications. The dataset includes customer attributes, loan attributes, and outcomes related to payment     behaviour, such as:

  Customers with payment difficulties (e.g., late payments beyond a certain threshold).
  Customers who paid their instalments on time.
  Loan Outcomes in the Dataset:
  Approved: The loan application was approved.
  Cancelled: Customer cancelled the application.
  Refused: The loan application was rejected.
  Unused Offer: The loan was approved but unused.

Business Objectives:
The primary aim of the project is to identify patterns and key factors that influence loan defaults. These insights can help the company:
  Deny loans to high-risk applicants.
  Reduce loan amounts for borderline applicants.
  Adjust interest rates for risky applicants.

* Project Workflow
1. Exploratory Data Analysis (EDA):
   
A. Handling Missing Data:
  Identified and visualized missing data across variables.
  Imputed missing values using statistical methods (e.g., mean or median).
  Graph Used: Bar chart to show the proportion of missing values per variable.

B. Identifying Outliers:
  Detected outliers in numerical variables using interquartile range (IQR) and statistical thresholds.
  Visualized outliers using box plots for better insights.
  Graph Used: Box plots to highlight the outliers in key variables.

C. Addressing Data Imbalance:
  Assessed class imbalance in the dataset, especially for binary classification problems.
  Calculated proportions and visualized the distribution of target variables.
  Graph Used: Pie charts and bar charts to depict class imbalance.

D. Univariate, Segmented Univariate, and Bivariate Analysis:
  Performed univariate analysis to study individual variable distributions.
  Conducted segmented univariate and bivariate analyses to compare variable distributions across different scenarios.
  Graphs Used: Histograms, stacked bar charts, grouped bar charts, and scatter plots.

E. Correlation Analysis:
  Analyzed correlations between features and the target variable for different scenarios.
  Identified top indicators of loan default based on correlation coefficients.
  Graph Used: Heatmaps to visualize the strongest correlations.

2. Machine Learning (ML) Models

To predict loan defaults, the following machine learning models were trained and evaluated:

A. Logistic Regression:
  A baseline linear model was used for binary classification.
  Achieved 91.5% accuracy on the test data.
B. Random Forest Classifier:
  An ensemble model providing robust predictions.
  Achieved 91.4% accuracy, with feature importance analysis to highlight key predictors.
C. Evaluation Metrics:
  Accuracy was calculated for both models.
  Confusion matrices were generated to evaluate prediction performance.
D. Feature Importance:
  Visualized the importance of features in predicting loan defaults using bar charts.

* Results and Insights
    Identified patterns in customer behaviour and loan attributes influencing defaults.
    Determined the most significant variables, such as income level, loan amount, and credit history.
    Built robust predictive models with over 91% accuracy to assist in decision-making.

* Tools and Technologies
  Languages: Python
  Libraries: pandas, numpy, matplotlib, scikit-learn
  Visualization Tools: Matplotlib and Seaborn for Python; Excel for additional analysis and visualizations.

* Usage Instructions
  Clone the Repository:
    git clone https://github.com/your-repo-name/bank-defaulter-prediction.git
    cd bank-defaulter-prediction

Install Dependencies:
  pip install -r requirements.txt

Run the Notebook:
  Open the provided Jupyter Notebook and execute the cells to perform EDA and train the machine learning models.

* Analyze Results:
  Review the visualizations and results generated during EDA.
  Evaluate the predictive performance of the models.

* Visualizations
The project includes various visualizations to enhance interpretability:
  Bar Charts: Proportion of missing values, class imbalance, and feature importance.
  Box Plots: Outlier detection.
  Heatmaps: Correlation analysis.
  Scatter Plots: Bivariate relationships.

* Business Impact
This project enables financial institutions to:
  Reduce default rates by identifying high-risk applicants.
  Increase profitability by optimizing loan approval strategies.
  Enhance customer trust with fair and data-driven decisions.
