The Code used is (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

# CodeClauseInternship_WineQualityPrediction

This code appears to be an analysis of a dataset related to wine quality. It involves data preprocessing, machine learning model selection, evaluation, and improvement. Here's a summary of the key steps and actions performed in the code:

1. **Data Loading and Exploration:**
   - Reads a CSV file named 'WineQT.csv' into a Pandas DataFrame.
   - Checks basic information about the dataset, including data types and the absence of null values.
   - Visualizes the distribution of wine quality using a count plot.

2. **Data Splitting and Scaling:**
   - Splits the data into input features (x) and the target variable (y).
   - Splits the data into training and testing sets using a 75%-25% split ratio.
   - Applies standard scaling to the input features (x_train and x_test) using the StandardScaler from scikit-learn.

3. **Linear Regression Model:**
   - Fits a Linear Regression model to the training data.
   - Predicts wine quality on the test data.
   - Calculates evaluation metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) for model assessment.

4. **Support Vector Regressor (SVR) Model:**
   - Fits an SVR model with a radial basis function (RBF) kernel to the training data.
   - Predicts wine quality on the test data.
   - Calculates evaluation metrics (MAE, MSE, RMSE, R²) for model assessment.

5. **Decision Tree Regressor Model:**
   - Fits a Decision Tree Regressor to the training data.
   - Predicts wine quality on the test data.
   - Calculates evaluation metrics (MAE, MSE, RMSE, R²) for model assessment.

6. **K-Fold Cross-Validation:**
   - Performs K-Fold Cross-Validation for Linear Regression and SVR models.
   - Computes Mean Squared Error (MSE) scores for each fold.
   - Reports the average MSE score for model assessment.

7. **Model Improvement (SVR):**
   - Repeats the SVR model selection and hyperparameter tuning process, now using an 80%-20% train-test split.
   - Performs GridSearchCV to find the best hyperparameters for the SVR model.
   - Evaluates the model using cross-validation on the training set and reports the average MSE score.
   - Calculates MSE on the test set using the improved SVR model.

The code demonstrates a thorough analysis of the wine quality dataset, including the use of multiple regression models and cross-validation for model evaluation and improvement. It provides insights into selecting the best-performing model and hyperparameters for the given task.
