# Census-Based-Income-Classification-Using-Machine-Learning
This project focuses on building an end-to-end Machine Learning pipeline using the Adult Income dataset (available on Kaggle). The objective is to classify whether an individual earns more or less than $50K per year by applying structured data preprocessing, feature engineering, feature selection, and model optimization techniques.

🚀 Project Overview
This project demonstrates a complete machine learning workflow, including:
Data profiling and preprocessing
Outlier treatment using IQR-based Winsorization
Feature engineering (Age Groups, Work Intensity)
Log transformations for skewed numerical variables
Hybrid encoding strategies (One-Hot + Label Encoding)
Anomaly detection with Isolation Forest
Feature selection using Mutual Information
Training and tuning ML models using GridSearchCV
Final evaluation using accuracy and classification metrics
The XGBoost model achieved an accuracy of approximately 81% on the test dataset.

📊 Dataset
Adult Census Income Dataset
Kaggle Link: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
The dataset includes 32,561 records describing demographic and employment-related attributes.

🧹 1. Data Preprocessing
Verified datatypes and validated dataset integrity
Removed duplicate records
Applied IQR-based Winsorization to stabilize high numeric values
Prepared categorical features for encoding
These steps improved overall data consistency and model performance.

🧩 2. Feature Engineering
Key engineered features include:
Age Groups (representing life-stage cohorts)
Work Intensity (categorizing work hours per week)
Additionally, log1p transformations were applied to handle heavy skew in capital_gain and capital_loss.

🔤 3. Encoding Strategy
A hybrid encoding approach was used:
One-Hot Encoding for low-cardinality categorical variables
Label Encoding for high-cardinality fields such as occupation and native_country
This kept the feature space efficient and model-friendly.

🔍 4. Feature Selection & Anomaly Detection
Applied Isolation Forest to detect and remove anomalies, improving signal quality
Used Mutual Information to quantify feature importance and identify key predictors such as:
Relationship
Marital Status
Education
Age
Hours Per Week
This selection process enhanced the model’s learning efficiency.

🤖 5. Model Development & Optimization
The following models were trained and evaluated:
Logistic Regression
Random Forest
XGBoost

Using GridSearchCV, hyperparameters such as tree depth, learning rate, subsampling, and regularization were tuned.
🏆 Best Result: XGBoost with ~81% accuracy.

📦 Technologies Used
Python
Pandas, NumPy
Scikit-learn
XGBoost
Matplotlib, Seaborn
Jupyter Notebook
