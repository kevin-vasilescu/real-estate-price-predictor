# Real Estate Price Predictor

A linear regression model to predict house prices based on neighborhood and housing characteristics. Built as a learning project to understand classical ML workflows.

## Motivation

I wanted to build a real machine learning project that goes through the full pipeline: exploratory data analysis, model training, evaluation, and making predictions. Price prediction is a great beginner project because it's practical (you actually *do* predict house prices in the real world) and linear regression is interpretable—you can see which features matter most.

## Problem

Given housing features (number of rooms, neighborhood demographics, school quality, etc.), predict the median house price in thousands of dollars.

## Dataset

Using the **Boston Housing Dataset** (modified version from sklearn).

- **Samples**: 506 houses
- **Features**: 7 selected features from the original 13
  - `RM`: Average number of rooms per dwelling
  - `LSTAT`: Percentage of lower status population
  - `PTRATIO`: Pupil-teacher ratio
  - `AGE`: Percentage of buildings built before 1940
  - `RAD`: Accessibility to radial highways (1-24)
  - `TAX`: Full-value property tax rate per $10K
  - `INDUS`: Percentage of non-retail business acres
- **Target**: `MEDV` - Median home value in $1000s

### Why this dataset?

- Small enough to understand what's happening
- No missing values (clean)
- Real-world problem
- Good for linear regression (continuous target, no extreme outliers)

## How It Works

**Model**: Linear Regression with feature scaling

Linear regression fits a line (hyperplane in higher dimensions) through the data by minimizing prediction error. The features are standardized so that each contributes fairly to the model.

**Features used**: Chose 7 features that seemed most important (rooms, demographics, school ratio, tax).

**Training**: 80% of data, test: 20%

## Results

```
Test RMSE: $4.79K
Test MAE:  $3.71K
Test R²:   0.6584
```

**What this means**:
- On average, predictions are off by ~$3,700
- The model explains ~66% of the variance in prices
- Not perfect, but reasonable for a simple linear model

### Feature Importance (Coefficients)

After standardization, these coefficients show relative importance:

```
RM        (rooms):     4.827   → More rooms = higher price (makes sense)
LSTAT     (lower %):  -3.760   → More lower status = lower price
PTRATIO   (teacher ratio): -1.042  → Higher class sizes = lower price
TAX              -0.917   → Higher taxes = lower price
INDUS            -0.621   → More industrial = lower price
AGE              -0.545   → Older buildings = lower price
RAD              -0.231   → Highway access = slight negative (probably collinear)
```

## Limitations

- **Only 7 features**: The original dataset has 13. More features would likely improve results.
- **Linear assumption**: Prices might have non-linear relationships (e.g., a house with 4 rooms isn't 4x cheaper than 1 room).
- **Outdated data**: The Boston Housing dataset is from the 1970s-80s. Modern prices have changed.
- **No interaction terms**: Didn't account for feature interactions (e.g., rooms × location).
- **R² of 0.66**: Still leaving ~34% of variance unexplained.

## Future Improvements

- Try more features (all 13 from the original dataset)
- Test other models: Random Forest, Gradient Boosting, Ridge/Lasso regression
- Feature engineering: polynomial features, interaction terms
- Cross-validation for more robust evaluation
- Visualize: predicted vs actual prices, residuals

## Setup & Run

### Requirements
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train_model.py
```

Outputs training/test metrics and saves the model.

### Make predictions
```bash
python make_predictions.py
```

Makes predictions on a few example houses.

### Project structure
```
.
├── train_model.py          # Train and evaluate
├── make_predictions.py     # Use trained model
├── data/
│   └── housing.csv        # Dataset
├── models/
│   ├── model.pkl          # Trained model (after running train_model.py)
│   └── scaler.pkl         # Feature scaler (after running train_model.py)
├── requirements.txt
├── .gitignore
└── README.md
```

## Reflection

This was a good first ML project. Linear regression is simple enough to understand completely, but still teaches important lessons about:
- Data preprocessing (handling missing values, scaling)
- Train/test splits
- Evaluating with multiple metrics (RMSE, MAE, R²)
- Feature importance
- When simple models are "good enough"

Next, I'd like to try a more complex dataset and compare multiple models to see which performs best.
