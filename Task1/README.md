# Titanic - Machine Learning from Disaster 🚢
This repository contains a Data Science project that predicts passenger survival on the Titanic using Machine Learning. This is a classic predictive modeling project based on the popular Kaggle challenge.
## 📌 Project Overview
The objective of this project is to analyze the Titanic passenger dataset and build a classification model to predict whether a passenger survived the shipwreck or not based on features like age, gender, socio-economic class, fare, and cabin.
## 📊 Dataset Description
The dataset used in this project is divided into two parts:
* `train.csv`: Contains the training data along with the target variable (`Survived`).
* `test.csv`: Contains the test data where the target variable needs to be predicted.
### Key Features:
* `PassengerId`: Unique ID for each passenger.
* `Survived`: Survival status (0 = No, 1 = Yes). **(Target Variable)**
* `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
* `Name`: Name of the passenger.
* `Sex`: Gender of the passenger.
* `Age`: Age in years.
* `SibSp`: Number of siblings/spouses aboard.
* `Parch`: Number of parents/children aboard.
* `Ticket`: Ticket number.
* `Fare`: Passenger fare.
* `Cabin`: Cabin number.
* `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
## 🛠️ Project Workflow
1. **Exploratory Data Analysis (EDA):** Visualizing data distributions, analyzing correlations, and identifying patterns (e.g., survival rates based on gender and class).
2. **Data Preprocessing:** * Handling missing values in `Age`, `Cabin`, and `Embarked`.
   * Encoding categorical variables (`Sex`, `Embarked`) into numerical formats.
   * Feature engineering (e.g., creating a `FamilySize` feature).
3. **Model Building:** Training various machine learning algorithms such as:
   * Logistic Regression
   * Random Forest Classifier
   * Decision Tree Classifier
4. **Model Evaluation:** Evaluating model performance using accuracy scores, confusion matrices, and classification reports.
## 💻 Tech Stack Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
## 📈 Key Insights & Results
* **Gender:** Female passengers had a significantly higher survival rate compared to male passengers.
* **Class:** Passengers in the 1st class (`Pclass = 1`) had a much higher probability of survival than those in the 3rd class.
* **Best Model:** *[Tip: Add your best performing model and its accuracy score here, e.g., "The Random Forest Classifier achieved the highest accuracy of 82%."]*
## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
