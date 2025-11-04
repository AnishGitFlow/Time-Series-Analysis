import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Load cleaned data
df_Anish = pd.read_csv('cleaned_timeseries.csv', index_col='date', parse_dates=True)

print("="*60)
print("LAB 4: STATIONARITY TRANSFORMATION TECHNIQUES")
print("="*60)

# Store original series for comparison
original_series = df_Anish['value'].copy()

# Q1: First Order Differencing
print("\nQ1. First Order Differencing")
print("="*60)

first_diff = df_Anish['value'].diff().dropna()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df_Anish.index, df_Anish['value'], color='steelblue', linewidth=1)
axes[0].set_title('Original Series', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Value', fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(first_diff.index, first_diff, color='orange', linewidth=1)
axes[1].set_title('After First Order Differencing', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=10)
axes[1].set_ylabel('Differenced Value', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab4_first_differencing.png', dpi=300, bbox_inches='tight')
plt.show()

# Test stationarity
ADF_diff1 = adfuller(first_diff, autolag='AIC')
print(f"\nADF Test after First Differencing:")
print(f"ADF Statistic: {ADF_diff1[0]:.6f}")
print(f"p-value: {ADF_diff1[1]:.6f}")
print(f"Inference: {'STATIONARY' if ADF_diff1[1] <= 0.05 else 'NON-STATIONARY'}")

print("\nQ1a. Comment:")
if ADF_diff1[1] <= 0.05:
    print("Stationarity is IMPROVED. The differenced series is stationary.")
    print("The trend has been removed, making the series fluctuate around zero.")
else:
    print("Stationarity needs further transformation.")

# Q2: Seasonal Differencing
print("\n" + "="*60)
print("Q2. Seasonal Differencing")
print("="*60)

seasonal_period = 7  # Assuming daily data with yearly seasonality
seasonal_diff = df_Anish['value'].diff(seasonal_period).dropna()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df_Anish.index, df_Anish['value'], color='steelblue', linewidth=1)
axes[0].set_title('Original Series', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Value', fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(seasonal_diff.index, seasonal_diff, color='green', linewidth=1)
axes[1].set_title(f'After Seasonal Differencing (lag={seasonal_period})', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=10)
axes[1].set_ylabel('Seasonal Differenced Value', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab4_seasonal_differencing.png', dpi=300, bbox_inches='tight')
plt.show()

ADF_seas = adfuller(seasonal_diff, autolag='AIC')
print(f"ADF Test after Seasonal Differencing:")
print(f"ADF Statistic: {ADF_seas[0]:.6f}")
print(f"p-value: {ADF_seas[1]:.6f}")
print(f"Inference: {'STATIONARY' if ADF_seas[1] <= 0.05 else 'NON-STATIONARY'}")

# Q3: Log and Box-Cox Transformation
print("\n" + "="*60)
print("Q3. Log and Box-Cox Transformation")
print("="*60)

# Ensure positive values
df_Anish_positive = df_Anish['value'] - df_Anish['value'].min() + 1

# Log Transformation
log_transform = np.log(df_Anish_positive)

# Box-Cox Transformation
boxcox_transform, lambda_param = boxcox(df_Anish_positive)
print(f"Optimal Lambda for Box-Cox: {lambda_param:.4f}")

# Plot Before and After
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Original
axes[0, 0].plot(df_Anish.index, df_Anish_positive, color='steelblue', linewidth=1)
axes[0, 0].set_title('Original Series', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Value', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df_Anish_positive, bins=50, color='steelblue', edgecolor='black')
axes[0, 1].set_title('Distribution - Original', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Value', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Log Transform
axes[1, 0].plot(df_Anish.index, log_transform, color='orange', linewidth=1)
axes[1, 0].set_title('Log Transformed', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Log(Value)', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(log_transform, bins=50, color='orange', edgecolor='black')
axes[1, 1].set_title('Distribution - Log Transform', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Log(Value)', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# Box-Cox Transform
axes[2, 0].plot(df_Anish.index, boxcox_transform, color='green', linewidth=1)
axes[2, 0].set_title(f'Box-Cox Transformed (λ={lambda_param:.2f})', 
                     fontsize=11, fontweight='bold')
axes[2, 0].set_ylabel('Box-Cox Value', fontsize=9)
axes[2, 0].set_xlabel('Date', fontsize=9)
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].hist(boxcox_transform, bins=50, color='green', edgecolor='black')
axes[2, 1].set_title('Distribution - Box-Cox Transform', fontsize=11, fontweight='bold')
axes[2, 1].set_xlabel('Box-Cox Value', fontsize=9)
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab4_transformations.png', dpi=300, bbox_inches='tight')
plt.show()

# Test transformations with differencing
log_diff = log_transform.diff().dropna()
boxcox_diff = pd.Series(boxcox_transform, index=df_Anish.index).diff().dropna()

ADF_log = adfuller(log_diff, autolag='AIC')
ADF_boxcox = adfuller(boxcox_diff, autolag='AIC')

print("\nQ3b. Effectiveness Comparison:")
print("-" * 60)
print(f"Log Transform + Differencing:")
print(f"  ADF p-value: {ADF_log[1]:.6f}")
print(f"  Variance: {log_diff.var():.6f}")

print(f"\nBox-Cox Transform + Differencing:")
print(f"  ADF p-value: {ADF_boxcox[1]:.6f}")
print(f"  Variance: {boxcox_diff.var():.6f}")

if ADF_boxcox[1] < ADF_log[1]:
    print("\nMost Effective: BOX-COX TRANSFORMATION")
    print("Reason: Lower p-value indicates stronger stationarity.")
    print("        Box-Cox optimally normalizes variance.")
else:
    print("\nMost Effective: LOG TRANSFORMATION")
    print("Reason: Lower p-value and simpler interpretation.")

# Q4: Comparison Table
print("\n" + "="*60)
print("Q4. Before vs After Transformation Comparison")
print("="*60)

# Run all tests
ADF_original = adfuller(original_series.dropna(), autolag='AIC')
kpss_original = kpss(original_series.dropna(), regression='c', nlags='auto')

kpss_diff1 = kpss(first_diff, regression='c', nlags='auto')
kpss_log = kpss(log_diff, regression='c', nlags='auto')
kpss_boxcox = kpss(boxcox_diff, regression='c', nlags='auto')

# Create comparison table
comparison = pd.DataFrame({
    'Transformation': [
        'Original',
        'First Differencing',
        'Log + Differencing',
        'Box-Cox + Differencing'
    ],
    'ADF Statistic': [
        ADF_original[0],
        ADF_diff1[0],
        ADF_log[0],
        ADF_boxcox[0]
    ],
    'ADF p-value': [
        ADF_original[1],
        ADF_diff1[1],
        ADF_log[1],
        ADF_boxcox[1]
    ],
    'ADF Result': [
        'Stationary' if ADF_original[1] <= 0.05 else 'Non-stationary',
        'Stationary' if ADF_diff1[1] <= 0.05 else 'Non-stationary',
        'Stationary' if ADF_log[1] <= 0.05 else 'Non-stationary',
        'Stationary' if ADF_boxcox[1] <= 0.05 else 'Non-stationary'
    ],
    'KPSS p-value': [
        kpss_original[1],
        kpss_diff1[1],
        kpss_log[1],
        kpss_boxcox[1]
    ],
    'KPSS Result': [
        'Stationary' if kpss_original[1] > 0.05 else 'Non-stationary',
        'Stationary' if kpss_diff1[1] > 0.05 else 'Non-stationary',
        'Stationary' if kpss_log[1] > 0.05 else 'Non-stationary',
        'Stationary' if kpss_boxcox[1] > 0.05 else 'Non-stationary'
    ]
})

print("\nComparison Table:")
print(comparison.to_string(index=False))

print("\n" + "-"*60)
print("Conclusion:")
print("-"*60)

# Determine best transformation
best_idx = comparison['ADF p-value'].idxmin()
best_transform = comparison.loc[best_idx, 'Transformation']

print(f"\nBest Transformation: {best_transform}")
print("This transformation achieved stationarity with:")
print(f"- Lowest ADF p-value: {comparison.loc[best_idx, 'ADF p-value']:.6f}")
print(f"- ADF Result: {comparison.loc[best_idx, 'ADF Result']}")
print(f"- KPSS Result: {comparison.loc[best_idx, 'KPSS Result']}")

print("\n" + "="*60)
print("LAB 4 COMPLETED")
print("="*60)