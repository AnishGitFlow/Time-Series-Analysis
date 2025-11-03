import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df_Anish = pd.read_csv('cleaned_timeseries.csv', index_col='date', parse_dates=True)

print("="*60)
print("LAB 6: MACHINE LEARNING FOR TIME SERIES FORECASTING")
print("="*60)

# Q1: Feature Engineering
print("\nQ1. Feature Engineering")
print("="*60)

# Create a copy for feature engineering
df_Anish_features = df_Anish.copy()

# Q1a: Lag Features
print("\nQ1a. Creating Lag Features")
df_Anish_features['lag1'] = df_Anish_features['value'].shift(1)
df_Anish_features['lag2'] = df_Anish_features['value'].shift(2)
df_Anish_features['lag3'] = df_Anish_features['value'].shift(3)

print("Created lag features: lag1, lag2, lag3")

# Q1b: Rolling Mean/Std
print("\nQ1b. Creating Rolling Statistics Features")
window = 7  # 7-day rolling window
df_Anish_features['rolling_mean'] = df_Anish_features['value'].rolling(window=window).mean()
df_Anish_features['rolling_std'] = df_Anish_features['value'].rolling(window=window).std()

print(f"Created rolling_mean and rolling_std with window={window}")

# Additional useful features
df_Anish_features['day_of_week'] = df_Anish_features.index.dayofweek
df_Anish_features['day_of_year'] = df_Anish_features.index.dayofyear
df_Anish_features['month'] = df_Anish_features.index.month

# Remove NaN values
df_Anish_features = df_Anish_features.dropna()

print(f"\nTotal features created: {len(df_Anish_features.columns) - 1}")
print("\nFeature columns:")
for col in df_Anish_features.columns:
    if col != 'value':
        print(f"  - {col}")

print(f"\nDataset shape after feature engineering: {df_Anish_features.shape}")
print(f"First few rows:")
print(df_Anish_features.head())

# Split into train and test
train_size = int(len(df_Anish_features) * 0.9)
train_data = df_Anish_features[:train_size]
test_data = df_Anish_features[train_size:]

# Separate features and target
X_train = train_data.drop('value', axis=1)
y_train = train_data['value']
X_test = test_data.drop('value', axis=1)
y_test = test_data['value']

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Q2: Train ML Models
print("\n" + "="*60)
print("Q2. Training Machine Learning Models")
print("="*60)

# Model 1: Linear Regression
print("\nModel 1: Linear Regression")
print("-" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("Model trained successfully")
print(f"Model coefficients: {len(lr_model.coef_)} features")
print(f"Intercept: {lr_model.intercept_:.4f}")

# Model 2: Random Forest
print("\nModel 2: Random Forest Regressor")
print("-" * 60)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("Model trained successfully")
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Max depth: {rf_model.max_depth}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head().to_string(index=False))

# Q2b: Plot Actual vs Predicted
print("\n" + "="*60)
print("Q2b. Plotting Actual vs Predicted Values")
print("="*60)

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Linear Regression
axes[0].plot(test_data.index, y_test.values, 
            label='Actual', color='black', linewidth=2, marker='o', markersize=3)
axes[0].plot(test_data.index, lr_predictions, 
            label='Predicted (Linear Regression)', color='blue', 
            linewidth=2, linestyle='--', marker='s', markersize=3)
axes[0].set_title('Linear Regression: Actual vs Predicted', 
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Value', fontsize=11)
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].plot(test_data.index, y_test.values, 
            label='Actual', color='black', linewidth=2, marker='o', markersize=3)
axes[1].plot(test_data.index, rf_predictions, 
            label='Predicted (Random Forest)', color='green', 
            linewidth=2, linestyle='--', marker='s', markersize=3)
axes[1].set_title('Random Forest: Actual vs Predicted', 
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Value', fontsize=11)
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab6_ml_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Create scatter plot for prediction accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression scatter
axes[0].scatter(y_test, lr_predictions, alpha=0.6, color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Values', fontsize=11)
axes[0].set_ylabel('Predicted Values', fontsize=11)
axes[0].set_title('Linear Regression: Prediction Accuracy', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest scatter
axes[1].scatter(y_test, rf_predictions, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Values', fontsize=11)
axes[1].set_ylabel('Predicted Values', fontsize=11)
axes[1].set_title('Random Forest: Prediction Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab6_prediction_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# Q3: Evaluate and Compare Models
print("\n" + "="*60)
print("Q3. Model Evaluation and Comparison")
print("="*60)

# Calculate metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_mape = mean_absolute_percentage_error(y_test, lr_predictions) * 100

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mape = mean_absolute_percentage_error(y_test, rf_predictions) * 100

# R-squared score
from sklearn.metrics import r2_score
lr_r2 = r2_score(y_test, lr_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Create comparison table
comparison_df_Anish = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'MAPE (%)': [lr_mape, rf_mape],
    'R² Score': [lr_r2, rf_r2]
})

print("\nModel Performance Comparison:")
print(comparison_df_Anish.to_string(index=False))

# Q3a: Which model gave better accuracy?
print("\n" + "="*60)
print("Q3a. Best Model Selection")
print("="*60)

if rf_rmse < lr_rmse:
    best_model = "Random Forest"
    best_rmse = rf_rmse
    best_mape = rf_mape
    best_r2 = rf_r2
else:
    best_model = "Linear Regression"
    best_rmse = lr_rmse
    best_mape = lr_mape
    best_r2 = lr_r2

print(f"\nBest Model: {best_model}")
print(f"RMSE: {best_rmse:.4f}")
print(f"MAPE: {best_mape:.2f}%")
print(f"R² Score: {best_r2:.4f}")

print("\nReason:")
if best_model == "Random Forest":
    print("Random Forest outperformed Linear Regression because:")
    print("- Can capture non-linear relationships in the data")
    print("- Ensemble approach reduces overfitting and variance")
    print("- Better handles complex temporal patterns and interactions")
    print("- Naturally handles feature interactions without explicit engineering")
else:
    print("Linear Regression outperformed Random Forest because:")
    print("- The underlying pattern is primarily linear")
    print("- Simpler model with lower variance on this dataset")
    print("- More interpretable with clear coefficient relationships")
    print("- Less prone to overfitting on this particular time series")

# Visualize performance comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Linear Regression', 'Random Forest']
rmse_values = [lr_rmse, rf_rmse]
mape_values = [lr_mape, rf_mape]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', color='steelblue')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, mape_values, width, label='MAPE (%)', color='coral')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12, color='steelblue')
ax2.set_ylabel('MAPE (%)', fontsize=12, color='coral')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig('lab6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Q3b: Strengths and Limitations
print("\n" + "="*60)
print("Q3b. Strengths and Limitations of ML Forecasting")
print("="*60)

print("\nStrengths of ML Forecasting:")
print("-" * 60)
print("1. FLEXIBLE FEATURE ENGINEERING:")
print("   Can incorporate multiple types of features (lags, rolling stats,")
print("   external variables) to capture complex patterns.")
print("\n2. NON-LINEAR RELATIONSHIPS:")
print("   Models like Random Forest can capture non-linear relationships")
print("   that traditional time series models might miss.")

print("\nLimitations of ML Forecasting:")
print("-" * 60)
print("1. REQUIRES EXTENSIVE FEATURE ENGINEERING:")
print("   Unlike ARIMA models that work directly with time series,")
print("   ML models need carefully crafted features, which requires")
print("   domain knowledge and can be time-consuming.")
print("\n2. MULTI-STEP AHEAD FORECASTING CHALLENGES:")
print("   ML models struggle with long-term forecasting because they rely")
print("   on lagged features. For multi-step forecasting, predictions from")
print("   previous steps must be used as inputs, causing error accumulation.")

# Additional insights
print("\n" + "-"*60)
print("Additional Considerations:")
print("-"*60)
print("• ML models work best with rich feature sets and large datasets")
print("• Traditional time series models (ARIMA) better for pure temporal patterns")
print("• Hybrid approaches combining both can often yield best results")
print(f"• In this experiment, {best_model} achieved {best_r2:.2%} variance explained")

# Save predictions for future reference
results_df_Anish = pd.DataFrame({
    'actual': y_test.values,
    'lr_predicted': lr_predictions,
    'rf_predicted': rf_predictions
}, index=test_data.index)

results_df_Anish.to_csv('lab6_ml_predictions.csv')
print("\n\nPredictions saved to 'lab6_ml_predictions.csv'")

print("\n" + "="*60)
print("LAB 6 COMPLETED")
print("="*60)
print("\nAll 6 labs completed successfully!")
print("Generated files:")
print("  - cleaned_timeseries.csv")
print("  - lab1_cleaned_series.png")
print("  - lab2_additive_decomposition.png")
print("  - lab2_multiplicative_decomposition.png")
print("  - lab3_rolling_stats.png")
print("  - lab4_first_differencing.png")
print("  - lab4_seasonal_differencing.png")
print("  - lab4_transformations.png")
print("  - lab5_acf_pacf.png")
print("  - lab5_all_forecasts.png")
print("  - lab6_ml_predictions.png")
print("  - lab6_prediction_accuracy.png")
print("  - lab6_model_comparison.png")
print("  - lab6_ml_predictions.csv")