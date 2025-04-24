🏡 House Price Prediction using Machine Learning

This project focuses on predicting median house values using the California housing dataset. It explores data preprocessing, feature engineering, and applying both Linear Regression and Random Forest models to build an effective prediction system.
🔍 Project Overview

The aim of this project is to:

    Explore and clean the housing dataset

    Perform data transformations and feature engineering

    Train and evaluate machine learning models

    Use hyperparameter tuning to optimize model performance

📁 Dataset

The dataset used is the California Housing dataset, which includes features such as:

    Total rooms

    Bedrooms

    Population

    Households

    Median income

    Ocean proximity

    Median house value (target)

⚙️ Technologies Used

    Python

    Pandas & NumPy for data manipulation

    Matplotlib & Seaborn for data visualization

    Scikit-learn for model building, evaluation, and hyperparameter tuning

📈 Models Trained
1. Linear Regression

    Used as a baseline model

    R² Score on test data: ~0.65

2. Random Forest Regressor

    Performs significantly better for non-linear relationships

    Hyperparameters tuned with GridSearchCV

    Final R² Score: ~0.80+ (after tuning)

🔄 Preprocessing Steps

    Removed null values

    One-hot encoding for categorical features (ocean_proximity)

    Log transformation for skewed features:

        total_rooms, total_bedrooms, population, households

    Created additional features:

        bedroom_ratio = total_bedrooms / total_rooms

        household_rooms = total_rooms / households

    Standardized features using StandardScaler

🔍 Hyperparameter Tuning

Used GridSearchCV on the Random Forest model to tune:

    n_estimators

    min_samples_split

    max_depth

📊 Results
Model	R² Score (Test Data)
Linear Regression	~0.65
Random Forest	~0.80+ (after tuning)
📂 How to Run

    Clone this repository:

git clone https://github.com/Mueez-lab/house-price-prediction.git
cd house-price-prediction

Install dependencies:

pip install -r requirements.txt

Run the main Python file:

    python house_price_prediction.py

📌 Future Improvements

    Try advanced models like Gradient Boosting or XGBoost

    Add cross-validation and more evaluation metrics

    Deploy model using a web app (e.g., Flask or Streamlit)

🙌 Acknowledgements

Thanks to the authors of the California Housing dataset and open-source libraries like scikit-learn, pandas, and matplotlib.
## 📬 Contact

Made by [**Mueez Zakir**](https://www.linkedin.com/in/mueezzakir/) — feel free to connect or reach out on LinkedIn!
