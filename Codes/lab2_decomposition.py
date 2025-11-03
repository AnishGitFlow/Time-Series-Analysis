import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load cleaned data from Lab 1
df_Anish = pd.read_csv('cleaned_timeseries.csv', index_col='date', parse_dates=True)

print("="*60)
print("LAB 2: TIME SERIES DECOMPOSITION")
print("="*60)

# Q1: Additive Decomposition
print("\nQ1. Additive Decomposition")
print("="*60)

additive_decomp = seasonal_decompose(df_Anish['value'], model='additive', period=7)

# Q1a: Plot components
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original
axes[0].plot(df_Anish.index, df_Anish['value'], color='steelblue')
axes[0].set_ylabel('Original', fontsize=11)
axes[0].set_title('Additive Decomposition', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(additive_decomp.trend.index, additive_decomp.trend, color='orange')
axes[1].set_ylabel('Trend', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Seasonality
axes[2].plot(additive_decomp.seasonal.index, additive_decomp.seasonal, color='green')
axes[2].set_ylabel('Seasonal', fontsize=11)
axes[2].grid(True, alpha=0.3)

# Residual
axes[3].plot(additive_decomp.resid.index, additive_decomp.resid, color='red')
axes[3].set_ylabel('Residual', fontsize=11)
axes[3].set_xlabel('Date', fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab2_additive_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# Q1b: Interpretations
print("\nQ1b. Interpretations from Additive Decomposition:")
print("1. The trend component shows a steady upward trajectory, indicating")
print("   consistent growth in the series over the observed period.")
print("2. The seasonal component reveals regular, repeating patterns with")
print("   consistent amplitude, suggesting stable seasonal effects throughout the series.")

# Q2: Multiplicative Decomposition
print("\n" + "="*60)
print("Q2. Multiplicative Decomposition")
print("="*60)

# Ensure no zeros or negative values for multiplicative
df_Anish_positive = df_Anish.copy()
df_Anish_positive['value'] = df_Anish_positive['value'] - df_Anish_positive['value'].min() + 1

multiplicative_decomp = seasonal_decompose(df_Anish_positive['value'], 
                                          model='multiplicative', period=7)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original
axes[0].plot(df_Anish_positive.index, df_Anish_positive['value'], color='steelblue')
axes[0].set_ylabel('Original', fontsize=11)
axes[0].set_title('Multiplicative Decomposition', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(multiplicative_decomp.trend.index, multiplicative_decomp.trend, color='orange')
axes[1].set_ylabel('Trend', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Seasonality
axes[2].plot(multiplicative_decomp.seasonal.index, multiplicative_decomp.seasonal, color='green')
axes[2].set_ylabel('Seasonal', fontsize=11)
axes[2].grid(True, alpha=0.3)

# Residual
axes[3].plot(multiplicative_decomp.resid.index, multiplicative_decomp.resid, color='red')
axes[3].set_ylabel('Residual', fontsize=11)
axes[3].set_xlabel('Date', fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab2_multiplicative_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# Q2a: Comparison
print("\nQ2a. Additive vs Multiplicative Comparison:")
print("-" * 60)

# Calculate variance of residuals
add_resid_var = additive_decomp.resid.var()
mult_resid_var = multiplicative_decomp.resid.var()

print(f"Additive Residual Variance: {add_resid_var:.4f}")
print(f"Multiplicative Residual Variance: {mult_resid_var:.4f}")

if add_resid_var < mult_resid_var:
    print("\nBest Fit: ADDITIVE")
    print("Reason: The additive model has lower residual variance, indicating")
    print("        better fit. Also, the seasonal amplitude remains constant,")
    print("        which is characteristic of additive seasonality.")
else:
    print("\nBest Fit: MULTIPLICATIVE")
    print("Reason: The multiplicative model has lower residual variance.")
    print("        This suggests seasonal variations increase proportionally with trend.")

# Q3: Explanation of components
print("\n" + "="*60)
print("Q3. Time Series Components Explained")
print("="*60)

print("\n1. TREND:")
print("   Definition: Long-term progressive change in the series mean over time.")
print("   Example: Increasing sales over years due to business growth.")

print("\n2. SEASONALITY:")
print("   Definition: Regular, periodic fluctuations occurring at fixed intervals.")
print("   Example: Higher ice cream sales in summer months every year.")

print("\n3. CYCLE:")
print("   Definition: Long-term oscillations with no fixed period, often spanning years.")
print("   Example: Economic business cycles of expansion and recession lasting 3-7 years.")

print("\n4. IRREGULAR (Residual):")
print("   Definition: Random, unpredictable variations remaining after other components.")
print("   Example: Sudden sales spike due to unexpected viral social media post.")

print("\n" + "="*60)
print("LAB 2 COMPLETED")
print("="*60)