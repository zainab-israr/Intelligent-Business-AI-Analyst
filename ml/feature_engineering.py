"""
Feature engineering for Intelligent Business AI Analyst.

Preprocesses numeric and categorical features for ML modeling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load dataset
data_path = 'data/business_data.csv'
df = pd.read_csv(data_path)

# Identify feature columns
numeric_features = ['sales_amount', 'units_sold', 'marketing_spend',
                    'customer_age', 'loyalty_score', 'visits_last_month',
                    'avg_purchase_value', 'discounts_used']

categorical_features = ['region', 'product_category']

target_column = 'target_success'

# Separate features and target
X = df[numeric_features + categorical_features]
y = df[target_column]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for future use
import joblib
os.makedirs('ml', exist_ok=True)
joblib.dump(preprocessor, 'ml/preprocessor.pkl')
print("Preprocessing complete. Preprocessor saved as ml/preprocessor.pkl")
