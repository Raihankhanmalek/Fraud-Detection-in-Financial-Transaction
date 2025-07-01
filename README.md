# Fraud Detection in Financial Transactions

This project demonstrates a machine learning approach to detecting fraudulent financial transactions using Python and popular data science libraries. The workflow includes data exploration, visualization, feature engineering, model training, and evaluation.

## Dataset
- **File:** `Fraud_detection_database.csv`
- The dataset contains transaction records with features such as transaction type, amount, and a label indicating whether the transaction is fraudulent.

## Notebook
- **File:** `Fraud_detection.ipynb`
- The notebook walks through the following steps:
  1. **Data Loading & Exploration:**
     - Loads the dataset and explores its structure and summary statistics.
     - Visualizes categorical and numerical features using seaborn and matplotlib.
  2. **Feature Engineering:**
     - Encodes categorical variables using one-hot encoding.
     - Prepares features (`X`) and target (`y`) for modeling.
  3. **Model Training:**
     - Splits the data into training and test sets.
     - Trains multiple models: Logistic Regression, XGBoost, and Random Forest.
     - Evaluates models using ROC AUC score.
  4. **Model Evaluation:**
     - Displays confusion matrix for model performance visualization.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage
1. Place `Fraud_detection_database.csv` in the project directory.
2. Open and run `Fraud_detection.ipynb` in Jupyter Notebook or VS Code.
3. Follow the notebook cells to explore the data, train models, and evaluate results.

## Results
- The notebook prints training and validation ROC AUC scores for each model.
- Visualizations include feature distributions, correlation heatmaps, and confusion matrices.

## License
This project is for educational purposes only.