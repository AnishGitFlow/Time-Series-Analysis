import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Load cleaned data
df_Anish = pd.read_csv('cleaned_timeseries.csv', index_col='date', parse_dates=True)

print("="*60)
print("LAB 3: STATIONARITY TESTING")
print("="*60)

# Q1: Rolling Mean and Rolling Standard Deviation
print("\nQ1. Rolling Statistics")
print("="*60)

# Calculate rolling statistics
window = 30  # 30-day window
rolling_mean = df_Anish['value'].rolling(window=window).mean()
rolling_std = df_Anish['value'].rolling(window=window).std()

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_Anish.index, df_Anish['value'], color='steelblue', label='Original', linewidth=1)
ax.plot(rolling_mean.index, rolling_mean, color='red', 
        label=f'Rolling Mean ({window}-day)', linewidth=2)
ax.plot(rolling_std.index, rolling_std, color='green', 
        label=f'Rolling Std ({window}-day)', linewidth=2)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Rolling Mean and Standard Deviation', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab3_rolling_stats.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nQ1a. Comment on Stationarity:")
print("The plot indicates NON-STATIONARITY because:")
print("- Rolling mean shows an upward trend (not constant)")
print("- Rolling standard deviation shows variations over time")
print("- For a stationary series, both should remain relatively constant")

# Q2: Augmented Dickey-Fuller (ADF) Test
print("\n" + "="*60)
print("Q2. Augmented Dickey-Fuller (ADF) Test")
print("="*60)

ADF_result = adfuller(df_Anish['value'].dropna(), autolag='AIC')

print(f"ADF Statistic: {ADF_result[0]:.6f}")
print(f"p-value: {ADF_result[1]:.6f}")
print(f"Number of lags used: {ADF_result[2]}")
print(f"Number of observations: {ADF_result[3]}")
print("\nCritical Values:")
for key, value in ADF_result[4].items():
    print(f"  {key}: {value:.4f}")

print("\nInference:")
if ADF_result[1] <= 0.05:
    print("p-value ≤ 0.05: REJECT null hypothesis")
    print("The series is STATIONARY (according to ADF)")
else:
    print("p-value > 0.05: FAIL TO REJECT null hypothesis")
    print("The series is NON-STATIONARY (according to ADF)")

# Q3: KPSS Test
print("\n" + "="*60)
print("Q3. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
print("="*60)

kpss_result = kpss(df_Anish['value'].dropna(), regression='c', nlags='auto')

print(f"KPSS Statistic: {kpss_result[0]:.6f}")
print(f"p-value: {kpss_result[1]:.6f}")
print(f"Number of lags used: {kpss_result[2]}")
print("\nCritical Values:")
for key, value in kpss_result[3].items():
    print(f"  {key}: {value:.4f}")

print("\nInference:")
if kpss_result[1] <= 0.05:
    print("p-value ≤ 0.05: REJECT null hypothesis")
    print("The series is NON-STATIONARY (according to KPSS)")
else:
    print("p-value > 0.05: FAIL TO REJECT null hypothesis")
    print("The series is STATIONARY (according to KPSS)")

# Q4: Conclusion and Explanation
print("\n" + "="*60)
print("Q4. Final Conclusion and Importance of Both Tests")
print("="*60)

# Create summary table
summary = pd.DataFrame({
    'Test': ['ADF', 'KPSS'],
    'Null Hypothesis': ['Non-stationary', 'Stationary'],
    'p-value': [ADF_result[1], kpss_result[1]],
    'Result': [
        'Stationary' if ADF_result[1] <= 0.05 else 'Non-stationary',
        'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
    ]
})

print("\nTest Summary:")
print(summary.to_string(index=False))

# Determine final conclusion
ADF_stationary = ADF_result[1] <= 0.05
kpss_stationary = kpss_result[1] > 0.05

print("\nFinal Conclusion:")
if ADF_stationary and kpss_stationary:
    print("STATIONARY: Both tests confirm stationarity")
elif not ADF_stationary and not kpss_stationary:
    print("NON-STATIONARY: Both tests confirm non-stationarity")
else:
    print("INCONCLUSIVE: Tests give conflicting results (difference stationary)")

print("\n" + "-"*60)
print("Why Check Both ADF & KPSS?")
print("-"*60)
print("\n1. COMPLEMENTARY HYPOTHESES:")
print("   - ADF: Null = Non-stationary (looks for unit root)")
print("   - KPSS: Null = Stationary (looks for trend/level stationarity)")
print("   Testing both provides stronger evidence.")

print("\n2. DIFFERENT TYPES OF NON-STATIONARITY:")
print("   - ADF detects: Unit root non-stationarity")
print("   - KPSS detects: Trend or level non-stationarity")
print("   Using both helps identify the specific type.")

print("\n3. CONFIRMATORY ANALYSIS:")
print("   - If both agree → Strong conclusion")
print("   - If they differ → Suggests difference-stationary series")
print("                      (stationary after differencing)")

print("\n4. AVOIDING TYPE I & II ERRORS:")
print("   - Using only one test risks false conclusions")
print("   - Both tests reduce risk of incorrect inference")

# Create visualization of decision matrix
print("\n" + "-"*60)
print("Decision Matrix:")
print("-"*60)
print("┌─────────────────┬──────────────┬──────────────────┐")
print("│ ADF Result      │ KPSS Result  │ Conclusion       │")
print("├─────────────────┼──────────────┼──────────────────┤")
print("│ Stationary      │ Stationary   │ Stationary       │")
print("│ Non-stationary  │ Non-stat.    │ Non-stationary   │")
print("│ Stationary      │ Non-stat.    │ Trend-stationary │")
print("│ Non-stationary  │ Stationary   │ Difference-stat. │")
print("└─────────────────┴──────────────┴──────────────────┘")

print("\n" + "="*60)
print("LAB 3 COMPLETED")
print("="*60)