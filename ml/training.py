"""
Train ML model for Intelligent Business AI Analyst.

Uses RandomForestClassifier to predict target_success.
Generates evaluation metrics and feature importance charts.
"""

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from feature_engineering import X_processed, y  # Import preprocessed data

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ml/model.pkl')
print("Model trained and saved as ml/model.pkl")

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc:.3f}")

# Plot feature importances (numeric + one-hot encoded categorical)
import numpy as np
feature_names_num = ['sales_amount', 'units_sold', 'marketing_spend',
                    'customer_age', 'loyalty_score', 'visits_last_month',
                    'avg_purchase_value', 'discounts_used']

# Get categorical feature names after one-hot encoding
preprocessor = joblib.load('ml/preprocessor.pkl')
cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(['region','product_category'])
feature_names = feature_names_num + list(cat_features)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
plt.title('Feature Importances')
plt.tight_layout()
os.makedirs('ml/charts', exist_ok=True)
plt.savefig('ml/charts/feature_importance.png')
plt.show()
print("Feature importance chart saved to ml/charts/feature_importance.png")
