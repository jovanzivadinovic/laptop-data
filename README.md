
# Laptop Price Analysis and Prediction Project

## Folder Structure

```
├── modules/
│   ├── cls_models.py         # Classification models (Random Forest, Gradient Boosting, Logistic Regression)
│   ├── data_clean.py         # Data cleaning and preprocessing functions
│   ├── exp_data_analysis.py  # Exploratory Data Analysis (EDA) and visualizations
│   ├── helpers.py            # Helper functions for data processing and visualizations
│   ├── kmeans_cluster.py     # Clustering using K-Means algorithm
│   ├── reg_models.py         # Regression models (Linear Regression, Random Forest, Gradient Boosting)
├── laptop_price.csv          # Dataset used for analysis (from Kaggle)
├── laptop_tasks.md           # Overview of tasks performed in the project
├── tasks.ipynb               # Jupyter Notebook for executing tasks interactively
├── requirements.txt          # List of Python dependencies for the modules
```

---

## What’s Implemented

1. **Data Cleaning (`data_clean.py`)**
   - Handles missing values, formats screen resolution, separates CPU/GPU details, and standardizes memory types.
   - Categorizes laptops into price ranges.

2. **Exploratory Data Analysis (EDA) (`exp_data_analysis.py`)**
   - Visualizes distributions of key features like RAM, CPU, screen size, and their relationships with price.
   - Provides statistical insights and correlation analysis.

3. **Regression Models (`reg_models.py`)**
   - Predicts laptop prices using Linear Regression, Random Forest, and Gradient Boosting.
   - Evaluates models with metrics like RMSE and MSE.

4. **Classification Models (`cls_models.py`)**
   - Classifies laptops into price brackets (e.g., Low, Medium, High).
   - Implements Random Forest, Gradient Boosting, and Logistic Regression models.

5. **Clustering (`kmeans_cluster.py`)**
   - Groups laptops into clusters based on specifications using K-Means.
   - Determines the optimal number of clusters with the Elbow Method.

6. **Feature Importance Analysis (`reg_models.py`)**
   - Uses SHAP values to identify key features influencing laptop prices.

7. **Recommendation System (`helpers.py`)**
   - Suggests laptops based on user preferences using cosine similarity.

8. **Helper Functions (`helpers.py`)**
   - Automates tasks like cleaning data, plotting trends, and encoding categorical variables.

---

## Jupyter Notebook: `tasks.ipynb`

The `tasks.ipynb` file serves as an interactive environment for:
- Running data cleaning and preprocessing functions.
- Performing EDA with visualizations.
- Training and evaluating machine learning models.
- Experimenting with clustering and recommendations.

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/username/laptop-price-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook tasks.ipynb
   ```

---

## Acknowledgements

- **Dataset Source:** [Laptop Price Dataset on Kaggle](https://www.kaggle.com/datasets/muhammetvarl/laptop-price?resource=download)