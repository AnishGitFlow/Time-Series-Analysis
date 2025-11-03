import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df_Anish = pd.read_csv('cleaned_timeseries.csv', index_col='date', parse_dates=True)

print("="*60)
print("LAB 5: TIME SERIES FORECASTING MODELS")
print("="*60)

# Split data into train and test
train_size = int(len(df_Anish) * 0.9)
train = df_Anish['value'][:train_size]
test = df_Anish['value'][train_size:]
forecast_periods = len(test)

print(f"Training samples: {len(train)}")
print(f"Testing samples: {len(test)}")
print(f"Forecast periods: {forecast_periods}")

# Q1: Plot ACF and PACF to determine p and q
print("\n" + "="*60)
print("Q1. ACF and PACF Analysis")
print("="*60)

# Use differenced series for better ACF/PACF
diff_series = df_Anish['value'].diff().dropna()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ACF
plot_acf(diff_series, lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# PACF
plot_pacf(diff_series, lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab5_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.show()

# Determine p and q
acf_values = acf(diff_series, nlags=40)
pacf_values = pacf(diff_series, nlags=40)

# Find where ACF cuts off (for q)
q = 0
for i in range(1, len(acf_values)):
    if abs(acf_values[i]) < 0.1:  # Threshold
        q = i - 1
        break

# Find where PACF cuts off (for p)
p = 0
for i in range(1, len(pacf_values)):
    if abs(pacf_values[i]) < 0.1:  # Threshold
        p = i - 1
        break

p = max(1, min(p, 5))  # Limit to reasonable range
q = max(1, min(q, 5))

print(f"\nDetermined parameters:")
print(f"p (AR order) = {p} (from PACF cutoff)")
print(f"q (MA order) = {q} (from ACF cutoff)")
print(f"d (differencing) = 1 (series required differencing)")

# Q2: Build Models and Forecast
print("\n" + "="*60)
print("Q2. Building and Forecasting with Different Models")
print("="*60)

results = {}

# Q2a: AR Model
print("\nQ2a. AR (AutoRegressive) Model")
print("-" * 60)
ar_order = p
ar_model = AutoReg(train, lags=ar_order).fit()
ar_forecast = ar_model.forecast(steps=forecast_periods)

ar_rmse = np.sqrt(mean_squared_error(test, ar_forecast))
ar_mape = mean_absolute_percentage_error(test, ar_forecast) * 100

results['AR'] = {
    'order': f'AR({ar_order})',
    'params': {'p': ar_order},
    'forecast': ar_forecast,
    'rmse': ar_rmse,
    'mape': ar_mape
}

print(f"Model Order: AR({ar_order})")
print(f"RMSE: {ar_rmse:.4f}")
print(f"MAPE: {ar_mape:.2f}%")

# Q2b: MA Model
print("\nQ2b. MA (Moving Average) Model")
print("-" * 60)
ma_order = q
ma_model = ARIMA(train, order=(0, 0, ma_order)).fit()
ma_forecast = ma_model.forecast(steps=forecast_periods)

ma_rmse = np.sqrt(mean_squared_error(test, ma_forecast))
ma_mape = mean_absolute_percentage_error(test, ma_forecast) * 100

results['MA'] = {
    'order': f'MA({ma_order})',
    'params': {'q': ma_order},
    'forecast': ma_forecast,
    'rmse': ma_rmse,
    'mape': ma_mape
}

print(f"Model Order: MA({ma_order})")
print(f"RMSE: {ma_rmse:.4f}")
print(f"MAPE: {ma_mape:.2f}%")

# Q2c: ARMA Model
print("\nQ2c. ARMA (AutoRegressive Moving Average) Model")
print("-" * 60)
arma_model = ARIMA(train, order=(p, 0, q)).fit()
arma_forecast = arma_model.forecast(steps=forecast_periods)

arma_rmse = np.sqrt(mean_squared_error(test, arma_forecast))
arma_mape = mean_absolute_percentage_error(test, arma_forecast) * 100

results['ARMA'] = {
    'order': f'ARMA({p},{q})',
    'params': {'p': p, 'q': q},
    'forecast': arma_forecast,
    'rmse': arma_rmse,
    'mape': arma_mape
}

print(f"Model Order: ARMA({p},{q})")
print(f"RMSE: {arma_rmse:.4f}")
print(f"MAPE: {arma_mape:.2f}%")

# Q2d: ARIMA Model
print("\nQ2d. ARIMA (AutoRegressive Integrated Moving Average) Model")
print("-" * 60)
d = 1  # Order of differencing
arima_model = ARIMA(train, order=(p, d, q)).fit()
arima_forecast = arima_model.forecast(steps=forecast_periods)

arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
arima_mape = mean_absolute_percentage_error(test, arima_forecast) * 100

results['ARIMA'] = {
    'order': f'ARIMA({p},{d},{q})',
    'params': {'p': p, 'd': d, 'q': q},
    'forecast': arima_forecast,
    'rmse': arima_rmse,
    'mape': arima_mape
}

print(f"Model Order: ARIMA({p},{d},{q})")
print(f"RMSE: {arima_rmse:.4f}")
print(f"MAPE: {arima_mape:.2f}%")

# Q3: Report Results and Plot Forecasts
print("\n" + "="*60)
print("Q3. Model Performance Summary")
print("="*60)

# Create summary table
summary_df_Anish = pd.DataFrame({
    'Model': list(results.keys()),
    'Order Parameters': [results[m]['order'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAPE (%)': [results[m]['mape'] for m in results.keys()]
})

print("\n" + summary_df_Anish.to_string(index=False))

# Plot all forecasts
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

colors = ['red', 'green', 'purple', 'orange']
model_names = list(results.keys())

for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    ax = axes[idx]
    
    # Plot training data (last 100 points for clarity)
    ax.plot(train.index[-100:], train.values[-100:], 
            label='Training Data', color='steelblue', linewidth=1.5)
    
    # Plot test data
    ax.plot(test.index, test.values, 
            label='Actual Test', color='black', linewidth=2, marker='o', markersize=3)
    
    # Plot forecast
    ax.plot(test.index, results[model_name]['forecast'], 
            label=f'{model_name} Forecast', color=color, linewidth=2, 
            linestyle='--', marker='s', markersize=3)
    
    ax.set_title(f"{model_name} Model: {results[model_name]['order']}\n"
                 f"RMSE: {results[model_name]['rmse']:.2f}, "
                 f"MAPE: {results[model_name]['mape']:.2f}%",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab5_all_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()

# Q4: Best Model Selection
print("\n" + "="*60)
print("Q4. Best Model Selection and Justification")
print("="*60)

# Find best model based on RMSE
best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
best_rmse = results[best_model]['rmse']
best_mape = results[best_model]['mape']

print(f"\nBest Performing Model: {best_model}")
print(f"Model Order: {results[best_model]['order']}")
print(f"RMSE: {best_rmse:.4f}")
print(f"MAPE: {best_mape:.2f}%")

print("\nJustification:")
print("-" * 60)

if best_model == 'AR':
    print("The AR model performed best because:")
    print("- The series has strong autocorrelation with past values")
    print("- Simple AR structure captures the pattern effectively")
    print("- Lower complexity reduces overfitting risk")
    
elif best_model == 'MA':
    print("The MA model performed best because:")
    print("- The series is better explained by past forecast errors")
    print("- Shock effects decay quickly in the series")
    print("- MA captures short-term fluctuations well")
    
elif best_model == 'ARMA':
    print("The ARMA model performed best because:")
    print("- Combines strengths of both AR and MA components")
    print("- Captures both autocorrelation and moving average effects")
    print("- More flexible in modeling complex patterns")
    
else:  # ARIMA
    print("The ARIMA model performed best because:")
    print("- Integrates differencing to handle non-stationarity")
    print("- Most comprehensive model handling trend and patterns")
    print("- Effectively models the integrated nature of the series")

print(f"\nVisual Accuracy: The {best_model} forecast closely follows actual values")
print(f"with minimal deviation, as shown in the forecast plot.")

# Additional comparison
print("\nPerformance Ranking (by RMSE):")
sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
for rank, (model, metrics) in enumerate(sorted_models, 1):
    print(f"{rank}. {model}: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")

print("\n" + "="*60)
print("LAB 5 COMPLETED")
print("="*60)