# 🛡️ Credit Card Fraud Detection

A machine learning project to identify fraudulent credit card transactions using classical and deep learning models. This project addresses the challenge of severe class imbalance and applies techniques like Focal Loss and ensemble methods.

## 🎯 Problem Statement

Fraudulent credit card transactions are rare but costly. The dataset is highly imbalanced, making traditional classifiers ineffective. The goal is to detect frauds accurately while minimizing false positives and handling data imbalance.

## 📦 Project Structure

Credit-Card-Fraud-Detection/
├── data/ # Training and testing datasets (ignored in Git)
├── models/ # Trained models (ignored in Git)
├── src/ # Source code for training and evaluation
├── utils/ # Utility functions
├── outputs/ # Metrics, logs, visualizations
├── README.md
├── requirements.txt
└── main.py # Entry point for running the pipeline

## 🤖 Models and Techniques

- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Neural Networks (PyTorch)
- Voting Classifier
- Focal Loss for class imbalance
- Hyperparameter tuning
- Confusion matrix & classification reports

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/khaled-hasan-hasan/Credit_Card_Fraud_Detection.git
   cd Credit_Card_Fraud_Detection

   python main.py --model logistic_regression

   python main.py --eval


---

### 📈 6. Results & Metrics

```markdown
## 📈 Results

| Model                | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.96     | 0.80      | 0.60   | 0.68     |
| Random Forest       | 0.98     | 0.87      | 0.72   | 0.79     |
| Neural Network      | 0.98     | 0.90      | 0.75   | 0.82     |
| Voting Classifier   | 0.98     | 0.91      | 0.76   | 0.83     |

## 🔧 Requirements

- Python 3.9+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- PyTorch

## 🧠 Lessons Learned

- How to handle highly imbalanced datasets
- Using Focal Loss with PyTorch
- Combining models using ensemble techniques
- Structuring a modular ML project


## 📮 Contact

Khaled Hassan  
📧 khaled.31033504@ics.tanta.edu.eg  
🔗 [LinkedIn](www.linkedin.com/in/khaled-hasan-751825203)

