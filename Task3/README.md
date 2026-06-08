# Iris Species Classification using Logistic Regression
This repository contains a complete Machine Learning pipeline to classify Iris flower species using Logistic Regression. 
## 📊 Key Highlights & Results
* **Exploratory Data Analysis:** Conducted feature relationship analysis via Pairplots to understand data distributions and separability.
* **Model Performance:** Achieved a stellar **96.67% accuracy score** on unseen test data.
* **Error Analysis:** The model classified 29 out of 30 test samples correctly, with only 1 minor misclassification between Versicolor and Virginica due to natural feature overlaps.
## 📁 Repository Structure
* `iris_classification.py`: Main Python script containing data loading, preprocessing, model training, and evaluation.
* `plots/`: Folder containing visualization outputs.
  * `pairplot.png` - Feature distribution grid.
  * `confusion_matrix.png` - Model evaluation heatmap.
## 🛠️ Tech Stack & Parameters
* **Language:** Python 3.14
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Model Configuration:** `LogisticRegression(max_iter=200, random_state=42)`
* **Data Split:** 80% Training (120 samples), 20% Testing (30 samples) with stratification.
