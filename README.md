# ✈️ Flight Price Prediction | MLP Term-2 Kaggle Assignment

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/mlp-term-2-2025-kaggle-assignment-1)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](https://github.com/)

> **Course:** Machine Learning Practices (MLP) - Term 2, 2025  
> **Institution:** IIT Madras BS Degree Programme  
> **Competition Duration:** June 20, 2025 - July 2, 2025

---

## 🎯 Competition Overview

This project involves predicting flight ticket prices based on various features such as airline, flight duration, departure time, number of stops, and booking patterns. The goal is to build a robust regression model that accurately estimates ticket prices for the test dataset.

### 🏆 Final Results
- **Private Leaderboard Score:** 0.965 (R² Score)
- **Best Score:** 0.965 (Version 2)
- **Evaluation Metric:** R² Score (Coefficient of Determination)

---

## 📊 Dataset Description

### Features
| Feature | Type | Description |
|---------|------|-------------|
| `airline` | Categorical | Name of the airline company |
| `flight` | Categorical | Flight code information |
| `source` | Categorical | Departure city |
| `departure` | Categorical | Time period of departure |
| `stops` | Categorical | Number of stops between cities |
| `arrival` | Categorical | Time period of arrival |
| `destination` | Categorical | Arrival city |
| `class` | Categorical | Seat class (Economy/Business) |
| `duration` | Numerical | Total travel time in hours |
| `days_left` | Numerical | Days between booking and departure |
| **`price`** | **Target** | **Ticket price (to predict)** |

### Dataset Statistics
- **Training Samples:** 40,000 records
- **Test Samples:** Varies
- **Missing Values:** Handled via imputation and removal strategies

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Treatment:**
  - Training data: Dropped rows with missing values
  - Test data: Forward fill imputation
- **Outlier Removal:** IQR method applied to `price` column
- **Duplicate Removal:** Ensured data integrity

### 2. Feature Engineering
- **Encoding:** Label Encoding for all categorical variables
- **Scaling:** StandardScaler applied to `duration` and `days_left`
- **Combined Encoding:** Train and test sets encoded together to maintain consistency

### 3. Model Development

#### Models Evaluated
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | 1649.01 | 3184.58 | **0.9804** |
| **XGBoost** | 1951.64 | 3411.98 | **0.9775** |
| Gradient Boosting | 2750.39 | 4590.81 | 0.9592 |
| Decision Tree | 1991.08 | 4643.89 | 0.9583 |
| Linear Regression | 4559.29 | 6885.08 | 0.9083 |
| Ridge Regression | 4559.91 | 6885.10 | 0.9083 |
| Lasso Regression | 4559.06 | 6885.10 | 0.9083 |
| KNN | 16900.78 | 22288.25 | 0.0393 |

#### Hyperparameter Tuning
**Final Model: XGBoost**
- Grid Search with 3-fold Cross-Validation
- Best Parameters:
  - `n_estimators`: 200
  - `max_depth`: 5
- **Final R² Score on Private Leaderboard:** 0.965

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.8+
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Model Selection: `GridSearchCV`

---

## 📈 Key Insights

1. **Tree-based models significantly outperformed linear models**, indicating non-linear relationships in the data
2. **Random Forest and XGBoost** showed the best performance with R² > 0.97
3. **Feature importance analysis** revealed:
   - `days_left` (booking timing) has strong predictive power
   - `airline` and `class` are critical pricing factors
   - `duration` correlates with price but shows complex interactions
4. **Outlier removal** improved model stability and generalization
5. **Hyperparameter tuning** provided marginal but important improvements

---

## 🚀 Reproducibility

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mlp-term2-2025-flight-price-prediction.git
cd mlp-term2-2025-flight-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Code
```bash
# Run the complete pipeline (if you create Python scripts)
python src/model_training.py

# Or open the Jupyter notebook
jupyter notebook notebooks/03_model_training_evaluation.ipynb
```

### Note on Data
Due to Kaggle competition rules, the dataset is not included in this repository. You can download it from the [competition page](https://www.kaggle.com/competitions/mlp-term-2-2025-kaggle-assignment-1/data).

---

## 📁 Project Structure

```
mlp-term2-2025-flight-price-prediction/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   └── README.md                     # Data download instructions
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/                              # Source code (optional)
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── results/
│   ├── model_comparison.csv
│   └── visualizations/
│
└── docs/
    ├── approach.md                   # Detailed methodology
    └── lessons_learned.md            # Reflections and learnings
```

---

## 🎓 Learning Outcomes

- Gained hands-on experience with **real-world regression problems**
- Mastered **end-to-end ML pipeline** from data preprocessing to model deployment
- Developed proficiency in **hyperparameter tuning** and model selection
- Improved understanding of **ensemble methods** (Random Forest, XGBoost, Gradient Boosting)
- Enhanced skills in **feature engineering** and handling categorical variables
- Learned best practices for **Kaggle competitions** and leaderboard submissions

---

## 🔮 Future Improvements

- [ ] Implement advanced feature engineering (polynomial features, interactions)
- [ ] Try deep learning models (Neural Networks)
- [ ] Ensemble different model predictions (stacking/blending)
- [ ] Perform more extensive hyperparameter optimization (Bayesian optimization)
- [ ] Add time-series features if booking patterns show temporal trends
- [ ] Experiment with feature selection techniques

---

## 📚 References

- [Kaggle Competition Link](https://www.kaggle.com/competitions/mlp-term-2-2025-kaggle-assignment-1)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- IIT Madras BS Programme - Machine Learning Practices Course Materials

---

## 👤 Author

**Your Name**  
IIT Madras BS Degree Programme | Data Science Student

- 📧 Email: aryanpatil1611@gmail.com
- 💼 LinkedIn: [Your LinkedIn](https://linkedin.com/in/aryanpatil97)
- 🏆 Kaggle: [Your Kaggle Profile](https://kaggle.com/aryansanjaypatil)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IIT Madras BS Programme Faculty for course guidance
- Kaggle community for inspiration and learning resources
- Fellow students for collaborative learning and discussions

---

**⭐ If you found this project helpful, please consider giving it a star!**
