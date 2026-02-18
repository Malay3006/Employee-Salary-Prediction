# Employee Salary Prediction

A machine learning project that predicts whether an employee's income exceeds $50K using the Adult Census Income Dataset. This project applies data preprocessing, feature engineering, and multiple classification algorithms to build an accurate predictive model.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [File Structure](#file-structure)
- [Requirements](#requirements)

## 🎯 Project Overview

This project focuses on building a machine learning pipeline to predict whether an individual's annual income exceeds $50,000. The workflow includes:

1. **Data Loading & Exploration** - Analyzing the structure and contents of the Adult Census Income dataset
2. **Data Cleaning** - Handling missing values and removing irrelevant features
3. **Data Preprocessing** - Encoding categorical variables and scaling numerical features
4. **Model Training** - Implementing multiple classification algorithms
5. **Model Evaluation** - Comparing accuracy and performance metrics

## 📊 Dataset

**Source:** Adult Census Income Dataset  
**Filename:** `adult.csv`  
**Size:** 32,561 records (original)

### Key Features:
- `age` - Age of the individual
- `workclass` - Employment status
- `education` - Highest level of education
- `marital.status` - Marital status
- `occupation` - Type of occupation
- `relationship` - Relationship status
- `race` - Race/ethnicity
- `sex` - Gender
- `capital.gain` - Capital gains
- `capital.loss` - Capital losses
- `hours.per.week` - Hours worked per week
- `native.country` - Country of origin
- `income` - Target variable (>50K or <=50K)

## ✨ Features

### Data Cleaning & Validation
- ✅ Identifies and handles missing values (represented as '?')
- ✅ Removes records with 'Preschool', '1st-4th', '5th-6th' education levels
- ✅ Filters out 'Never-worked' and 'Without-pay' employment categories
- ✅ Age-based filtering (17-75 years)

### Categorical Encoding
- Uses `LabelEncoder` to convert categorical variables into numerical format
- Encodes: workclass, marital status, occupation, relationship, race, native country, gender

### Feature Scaling
- Implements `MinMaxScaler` for normalizing numerical features
- Scales features to [0, 1] range for optimal model performance

### Model Selection
- **Logistic Regression** - Baseline linear classifier
- **K-Nearest Neighbors (KNN)** - Non-parametric classifier
- **Multi-Layer Perceptron (MLP)** - Neural network approach

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/employee-salary-prediction.git
cd employee-salary-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📖 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

2. Run cells sequentially to:
   - Load and explore the data
   - Clean and preprocess the dataset
   - Train multiple classification models
   - Evaluate model performance

3. View visualizations and predictions in the notebook output

## 🔧 Data Preprocessing

### Step 1: Data Cleaning
- Remove missing values indicated by '?'
- Drop irrelevant education categories (Preschool, 1st-4th, 5th-6th)
- Filter employment categories

### Step 2: Outlier Removal
- Age filtering: Keep records where 17 ≤ age ≤ 75
- Visualized with boxplots before and after filtering

### Step 3: Categorical Encoding
All categorical features are label-encoded:
```python
encoder = LabelEncoder()
data['workclass'] = encoder.fit_transform(data['workclass'])
data['sex'] = encoder.fit_transform(data['sex'])
# ... etc for other categorical columns
```

### Step 4: Feature Scaling
- MinMaxScaler normalizes features to [0, 1] range
- Applied to training data, then transformed on test data

### Step 5: Train-Test Split
- 80% training data
- 20% test data

## 🤖 Models Implemented

### 1. Logistic Regression
- Simple, interpretable linear classifier
- Good baseline for comparison

### 2. K-Nearest Neighbors (KNN)
- Non-parametric approach
- Flexible decision boundaries

### 3. Multi-Layer Perceptron (MLP)
- Neural network with hidden layers
- Captures complex patterns

## 📈 Results

Model performance is evaluated using:
- **Accuracy Score** - Overall correctness of predictions
- **Classification Report** - Precision, Recall, and F1-score

Models are compared to identify the best performer on the test dataset.

## 📁 File Structure

```
employee-salary-prediction/
│
├── README.md                 # Project documentation
├── main.ipynb               # Main Jupyter notebook with complete analysis
├── adult.csv                # Dataset
└── requirements.txt         # Python dependencies
```

## 📦 Requirements

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## 🔍 Key Insights

1. **Data Quality** - Dataset required significant cleaning (handling missing values, removing irrelevant categories)
2. **Feature Engineering** - Encoding categorical variables is crucial for model performance
3. **Model Comparison** - Different algorithms provide varying levels of accuracy
4. **Preprocessing Importance** - Proper scaling and normalization improve model convergence

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 📞 Contact

For questions or feedback, please feel free to reach out.

---

**Last Updated:** February 2026  
**Status:** ✅ Complete
