# Time Series Analysis - Complete Lab Guide

A comprehensive 6-lab series covering end-to-end time series analysis, from data preprocessing to advanced forecasting using statistical and machine learning models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Lab Descriptions](#lab-descriptions)
- [Running the Labs](#running-the-labs)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Key Learnings](#key-learnings)

---

## 🎯 Overview

This project contains 6 comprehensive labs that cover:
1. **Data Loading & Cleaning** - Handling missing values, outliers, and duplicates
2. **Time Series Decomposition** - Breaking down series into trend, seasonality, and residuals
3. **Stationarity Testing** - ADF and KPSS tests to check series properties
4. **Transformation Techniques** - Differencing, log, and Box-Cox transformations
5. **Classical Forecasting Models** - AR, MA, ARMA, and ARIMA models
6. **Machine Learning Forecasting** - Linear Regression and Random Forest with feature engineering

---

## 🔧 Prerequisites

### Python Version
- Python 3.7 or higher

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

---

## 📦 Installation

### Step 1: Clone or Download the Repository
```bash
# If using git
git clone https://github.com/AnishGitFlow/Time-Series-Analysis
cd time-series-analysis-labs

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Information

### Dataset: `time_series_retail_dataset.csv`

**Columns:**
- `Date` - Date of observation (YYYY-MM-DD format)
- `Sales` - Sales amount (target variable)
- `Promotion_Flag` - Binary flag for promotions (0/1)
- `Holiday_Flag` - Binary flag for holidays (0/1)
- `Temperature` - Temperature in degrees
- `Footfall` - Customer footfall count

**Sample Data:**
```csv
Date,Sales,Promotion_Flag,Holiday_Flag,Temperature,Footfall
2023-03-16,188.07,,0.0,30.32,560.0
2023-07-13,205.49,0.0,0.0,21.45,619.0
2023-09-29,238.54,0.0,0.0,22.69,721.0
```

### Dataset Requirements
- Minimum 365 rows recommended for seasonal analysis
- Date column must be in parseable date format
- Target variable (Sales) should be numeric
- Missing values are handled automatically

---

## 📚 Lab Descriptions

### Lab 1: Data Loading and Cleaning
**Objectives:**
- Load time series data into Python
- Convert date column to DateTime format
- Handle missing values using forward fill
- Detect and treat outliers using IQR method
- Check and remove duplicates
- Visualize cleaned series

**Key Outputs:**
- `cleaned_timeseries.csv` - Cleaned dataset
- `lab1_cleaned_series.png` - Time series plot

---

### Lab 2: Time Series Decomposition
**Objectives:**
- Perform additive decomposition
- Perform multiplicative decomposition
- Compare both decomposition methods
- Understand trend, seasonality, and residual components

**Key Outputs:**
- `lab2_additive_decomposition.png` - Additive components
- `lab2_multiplicative_decomposition.png` - Multiplicative components

**Key Concepts:**
- **Trend:** Long-term direction of the series
- **Seasonality:** Regular, periodic fluctuations
- **Residual:** Random variations

---

### Lab 3: Stationarity Testing
**Objectives:**
- Plot rolling statistics
- Perform ADF (Augmented Dickey-Fuller) test
- Perform KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
- Understand importance of both tests

**Key Outputs:**
- `lab3_rolling_stats.png` - Rolling mean and std deviation

**Key Tests:**
- **ADF Test:** H₀ = Series is non-stationary
- **KPSS Test:** H₀ = Series is stationary

---

### Lab 4: Transformation Techniques
**Objectives:**
- Apply first-order differencing
- Apply seasonal differencing
- Apply log transformation
- Apply Box-Cox transformation
- Compare effectiveness of transformations

**Key Outputs:**
- `lab4_first_differencing.png`
- `lab4_seasonal_differencing.png`
- `lab4_transformations.png`

**Transformation Goals:**
- Remove trend
- Stabilize variance
- Achieve stationarity

---

### Lab 5: Classical Time Series Models
**Objectives:**
- Plot ACF and PACF to determine model parameters
- Build AR, MA, ARMA, and ARIMA models
- Forecast next 12 periods
- Compare model performance

**Key Outputs:**
- `lab5_acf_pacf.png` - ACF and PACF plots
- `lab5_all_forecasts.png` - All model forecasts

**Models:**
- **AR(p):** Autoregressive model
- **MA(q):** Moving Average model
- **ARMA(p,q):** Combined AR and MA
- **ARIMA(p,d,q):** Integrated ARMA

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

### Lab 6: Machine Learning Forecasting
**Objectives:**
- Create lag features and rolling statistics
- Train Linear Regression model
- Train Random Forest model
- Compare ML vs traditional methods

**Key Outputs:**
- `lab6_ml_predictions.png` - Prediction plots
- `lab6_prediction_accuracy.png` - Scatter plots
- `lab6_model_comparison.png` - Performance comparison
- `lab6_ml_predictions.csv` - Prediction results

**Features Created:**
- Lag features (lag1, lag2, lag3)
- Rolling mean and std
- Time-based features (day, month, day of week)

---

## 🚀 Running the Labs

### Quick Start
```bash
# Ensure your dataset is in the same directory
# File name: time_series_retail_dataset.csv

# Run labs in sequence
python lab1_data_loading.py
python lab2_decomposition.py
python lab3_stationarity.py
python lab4_transformation.py
python lab5_time_series_models.py
python lab6_ml_forecasting.py
```

### Running Individual Labs
```bash
# Lab 1 must be run first as it creates cleaned_timeseries.csv
python lab1_data_loading.py

# Other labs can be run independently after Lab 1
python lab3_stationarity.py
```

### Running in Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Open and run each lab notebook sequentially
```

---

## 📁 Output Files

### Generated CSV Files
| File | Description |
|------|-------------|
| `cleaned_timeseries.csv` | Cleaned dataset after Lab 1 |
| `lab6_ml_predictions.csv` | ML model predictions |

### Generated Visualizations
| File | Lab | Description |
|------|-----|-------------|
| `lab1_cleaned_series.png` | 1 | Cleaned time series plot |
| `lab2_additive_decomposition.png` | 2 | Additive decomposition components |
| `lab2_multiplicative_decomposition.png` | 2 | Multiplicative decomposition |
| `lab3_rolling_stats.png` | 3 | Rolling mean and std deviation |
| `lab4_first_differencing.png` | 4 | First-order differencing result |
| `lab4_seasonal_differencing.png` | 4 | Seasonal differencing result |
| `lab4_transformations.png` | 4 | Log and Box-Cox transformations |
| `lab5_acf_pacf.png` | 5 | ACF and PACF plots |
| `lab5_all_forecasts.png` | 5 | All model forecasts comparison |
| `lab6_ml_predictions.png` | 6 | ML predictions vs actual |
| `lab6_prediction_accuracy.png` | 6 | Prediction scatter plots |
| `lab6_model_comparison.png` | 6 | Model performance bar chart |

---

## 🔍 Troubleshooting

### Common Issues

#### Issue 1: "File not found" error
```
FileNotFoundError: [Errno 2] No such file or directory: 'time_series_retail_dataset.csv'
```
**Solution:** Ensure the dataset file is in the same directory as the Python scripts.

#### Issue 2: "Module not found" error
```
ModuleNotFoundError: No module named 'statsmodels'
```
**Solution:** Install missing libraries:
```bash
pip install statsmodels
```

#### Issue 3: "cleaned_timeseries.csv not found" in Labs 2-6
**Solution:** Run Lab 1 first to generate the cleaned dataset.

#### Issue 4: Deprecation warnings
**Solution:** These are usually harmless. To suppress warnings, add at the top:
```python
import warnings
warnings.filterwarnings('ignore')
```

#### Issue 5: Memory issues with large datasets
**Solution:** 
- Reduce the dataset size by sampling
- Adjust the seasonal period in decomposition
- Reduce the number of trees in Random Forest

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Ensure all prerequisites are installed
3. Verify dataset format matches requirements
4. Check Python version compatibility

---

## 📈 Key Learnings

### Statistical Concepts
✅ Understanding stationarity and its importance  
✅ Decomposing time series into components  
✅ Choosing between additive and multiplicative models  
✅ Interpreting ACF and PACF plots  
✅ Selecting appropriate ARIMA parameters  

### Practical Skills
✅ Data cleaning for time series  
✅ Handling missing values and outliers  
✅ Feature engineering for ML models  
✅ Model evaluation and comparison  
✅ Visualizing time series data effectively  

### Model Comparison
| Aspect | Statistical Models | ML Models |
|--------|-------------------|-----------|
| **Interpretability** | High | Medium to Low |
| **Feature Engineering** | Minimal | Extensive |
| **Long-term Forecasting** | Better | Challenging |
| **Non-linear Patterns** | Limited | Excellent |
| **Training Time** | Fast | Slower |

---

## 📄 License

This project is created for educational purposes.

---

## 👥 Contributing

Contributions are welcome! If you find any issues or have suggestions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Contact

For questions or support, please create an issue in the repository.

---
**Happy Time Series Analysis! 📊📈**

---

*Last Updated: November 2024*
