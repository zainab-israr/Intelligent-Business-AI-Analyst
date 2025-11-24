import pandas as pd
import matplotlib.pyplot as plt
import os

def make_charts(df):
    os.makedirs('reports/assets', exist_ok=True)
    p1 = 'reports/assets/churn_hist.png'
    plt.figure(figsize=(6,4))
    df['pred_proba'].hist(bins=30)
    plt.title('Predicted Churn Probability')
    plt.savefig(p1)
    plt.close()

    p2 = 'reports/assets/salary_vs_proba.png'
    plt.figure(figsize=(6,4))
    df.boxplot(column='pred_proba', by='salary_band')
    plt.title('Predicted Prob by Salary Band')
    plt.suptitle('')
    plt.savefig(p2)
    plt.close()
    return [p1,p2]
