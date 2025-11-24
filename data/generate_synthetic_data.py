"""
Generate synthetic business dataset for Intelligent Business AI Analyst.

Creates a CSV file with:
- customer_id
- numeric business metrics
- categorical features
- feedback/review text
- target_success (binary)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Generate numeric features for 500 customers
X, y = make_classification(
    n_samples=500,
    n_features=8,          # numeric business metrics
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(X, columns=[
    'sales_amount', 'units_sold', 'marketing_spend',
    'customer_age', 'loyalty_score', 'visits_last_month',
    'avg_purchase_value', 'discounts_used'
])

# Add customer ID
df['customer_id'] = np.arange(1, len(df)+1)

# Add categorical features
df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))
df['product_category'] = np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], size=len(df))

# Add textual feedback / reviews
feedback_options = [
    'very satisfied with product',
    'neutral experience',
    'unsatisfied with delivery',
    'excited about the product',
    'angry about late shipment',
    'happy with customer support'
]
df['feedback'] = np.random.choice(feedback_options, size=len(df))

# Add target variable: 1 = successful transaction / high-value customer, 0 = otherwise
df['target_success'] = y

# Save CSV
df.to_csv('data/business_data.csv', index=False)
print("Synthetic dataset generated: data/business_data.csv")
print(df.head())
