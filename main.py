
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# 1. Load the config
with open('config/trainer_config.yml', 'r') as file:
    config = yaml.safe_load(file)

lda_config = config['trainer']['lda']
params = lda_config['parameters']

# 2. Load your data
# Replace with your actual dataset loading method
data = pd.read_csv('/home/khaled-hasan/Credit-Card-Fraud-Detection1/data/split/train.csv')  # You should already have this preprocessed
X = data.drop('Class', axis=1)
y = data['Class']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Build the LDA model
if params.get("solver") == "svd":
    lda_model = LinearDiscriminantAnalysis(solver='svd')
else:
    lda_model = LinearDiscriminantAnalysis(
        solver=params.get("solver", "lsqr"),
        shrinkage=params.get("shrinkage", None)
    )

# 5. Train and evaluate
lda_model.fit(X_train, y_train)
y_pred = lda_model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
