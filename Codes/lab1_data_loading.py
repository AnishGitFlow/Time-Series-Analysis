import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load the actual retail dataset
df_Anish = pd.read_csv('time_series_retail_dataset.csv')

# Rename columns for consistency with the rest of the code
df_Anish = df_Anish.rename(columns={'Date': 'date', 'Sales': 'value'})

# Keep only the date and value columns for time series analysis
df_Anish = df_Anish[['date', 'value']]

print("="*60)
print("LAB 1: DATA LOADING AND CLEANING")
print("="*60)

# Q1a: Display first and last 10 rows
print("\nQ1a. First 10 rows:")
print(df_Anish.head(10))
print("\nLast 10 rows:")
print(df_Anish.tail(10))

# Q1b: Convert date to DateTime and set as index
df_Anish['date'] = pd.to_datetime(df_Anish['date'])
df_Anish.set_index('date', inplace=True)
df_Anish.sort_values('date', inplace=True)
print("\nQ1b. Date column converted to DateTime and set as index")
print(df_Anish.info())

# Q2a: Check and handle missing values
print("\n" + "="*60)
print("Q2a. Missing Values Analysis")
print("="*60)
print(f"Number of missing values: {df_Anish['value'].isna().sum()}")
print(f"Percentage of missing values: {df_Anish['value'].isna().sum()/len(df_Anish)*100:.2f}%")

# Method: Forward fill (carry last observation forward)
# Justification: For time series, forward fill preserves temporal continuity
df_Anish['value'] = df_Anish['value'].fillna(method='ffill')
print("Method chosen: Forward Fill (ffill)")
print("Reason: Maintains temporal continuity by carrying last valid observation forward")
print(f"Missing values after treatment: {df_Anish['value'].isna().sum()}")

# Q2b: Detect and treat outliers
print("\n" + "="*60)
print("Q2b. Outlier Detection and Treatment")
print("="*60)

# Method: IQR (Interquartile Range)
Q1 = df_Anish['value'].quantile(0.25)
Q3 = df_Anish['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_Anish[(df_Anish['value'] < lower_bound) | (df_Anish['value'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

# Treat outliers by capping
df_Anish['value'] = df_Anish['value'].clip(lower=lower_bound, upper=upper_bound)
print("Method: IQR with capping")
print("Reason: IQR is robust to extreme values and capping preserves data points")

# Q2c: Check for duplicates
print("\n" + "="*60)
print("Q2c. Duplicate Detection")
print("="*60)
duplicates = df_Anish.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    df_Anish = df_Anish.drop_duplicates()
    print("Duplicates removed")
else:
    print("No duplicates found")

# Q3: Plot cleaned time series
print("\n" + "="*60)
print("Q3. Time Series Plot and Observations")
print("="*60)

plt.figure(figsize=(14, 6))
plt.plot(df_Anish.index, df_Anish['value'], linewidth=1, color='steelblue')
plt.title('Cleaned Time Series Data', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab1_cleaned_series.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nObservations:")
print("1. The series shows a clear upward trend over time with values increasing gradually.")
print("2. There appears to be seasonal/cyclical patterns repeating at regular intervals.")

# Save cleaned data
df_Anish.to_csv('cleaned_timeseries.csv')
print("\nCleaned data saved to 'cleaned_timeseries.csv'")
print("\n" + "="*60)
print("LAB 1 COMPLETED")
print("="*60)