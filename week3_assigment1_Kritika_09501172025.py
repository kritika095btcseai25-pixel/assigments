import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# PART A: Understanding the Dataset
# ============================================================

# Q1. Dataset Overview
df = pd.read_csv("agriculture_ml_dataset_cw.csv")

print("Q1. Shape:", df.shape[0], "rows,", df.shape[1], "columns")
print("Q1. Columns:", list(df.columns))
print("\nQ1. First 10 records:\n", df.head(10))

# Q2. Data Types and Missing Values
print("\nQ2. Data types:\n", df.dtypes)

missing = df.isnull().sum()
missing_cols = missing[missing > 0]
print("\nQ2. Missing values per column:\n", missing)
print("Q2. Columns with missing values:", list(missing_cols.index))

# Q3. Descriptive Statistics
numeric_df = df.select_dtypes(include=[np.number])
print("\nQ3. Summary statistics:\n", numeric_df.describe())
print("Q3. Feature with highest mean:", numeric_df.mean().idxmax())
print("Q3. Feature with highest std :", numeric_df.std().idxmax())

# ============================================================
# PART B: Exploratory Data Analysis (EDA)
# ============================================================

# Q4. Distribution Analysis
# Note: the brief asks for "fertilizer_kg", but in this dataset "Fertilizer"
# is categorical (Chemical/Organic/Mixed), not a kg amount. Using "Nitrogen"
# instead as the numeric nutrient-related feature.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flatten(), ["Rainfall", "Temperature", "Nitrogen", "Yield"]):
    ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Q5. Crop Type Analysis
crop_counts = df["Crop_Type"].value_counts()
print("\nQ5. Records per crop type:\n", crop_counts)

plt.figure(figsize=(7, 5))
crop_counts.plot(kind="bar", color="#55A868", edgecolor="black")
plt.title("Count of Records by Crop Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("Q5. Most frequent crop:", crop_counts.idxmax())

# Q6. Soil Type Analysis
soil_counts = df["Soil_Type"].value_counts()
print("\nQ6. Records per soil type:\n", soil_counts)

plt.figure(figsize=(7, 5))
soil_counts.plot(kind="bar", color="#C44E52", edgecolor="black")
plt.title("Count of Records by Soil Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("Q6. Most common soil type:", soil_counts.idxmax())

# Q7. Yield Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Yield"].dropna(), bins=30, color="#8172B2", edgecolor="white")
plt.title("Distribution of Yield")
plt.tight_layout()
plt.show()

Q1_, Q3_ = df["Yield"].quantile(0.25), df["Yield"].quantile(0.75)
IQR = Q3_ - Q1_
lower, upper = Q1_ - 1.5 * IQR, Q3_ + 1.5 * IQR
outliers = df[(df["Yield"] < lower) | (df["Yield"] > upper)]

print("Q7. Yield skewness:", round(df["Yield"].skew(), 3), "(near 0 -> roughly normal)")
print("Q7. Outliers outside [", round(lower, 2), ",", round(upper, 2), "]:", len(outliers))

# Q8. Scatter Plot Analysis
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(df["Rainfall"], df["Yield"], alpha=0.4, color="#4C72B0")
axes[0].set_title("Rainfall vs Yield")
axes[1].scatter(df["Nitrogen"], df["Yield"], alpha=0.4, color="#DD8452")
axes[1].set_title("Nitrogen vs Yield")
plt.tight_layout()
plt.show()

corr_rain = df["Rainfall"].corr(df["Yield"])
corr_nitro = df["Nitrogen"].corr(df["Yield"])
print("\nQ8. Correlation Rainfall-Yield:", round(corr_rain, 3))
print("Q8. Correlation Nitrogen-Yield:", round(corr_nitro, 3))
print("Q8. Stronger relationship with yield:", "Rainfall" if abs(corr_rain) > abs(corr_nitro) else "Nitrogen")

# Q9. Correlation Analysis
corr_matrix = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

top3 = corr_matrix["Yield"].drop("Yield").abs().sort_values(ascending=False).head(3)
print("\nQ9. Top 3 features correlated with Yield:\n", top3)

# Q10. Group-Based Analysis
avg_yield_crop = df.groupby("Crop_Type")["Yield"].mean().sort_values(ascending=False)
avg_yield_soil = df.groupby("Soil_Type")["Yield"].mean().sort_values(ascending=False)
print("\nQ10. Average yield by crop:\n", avg_yield_crop)
print("\nQ10. Average yield by soil:\n", avg_yield_soil)
print("Q10. Best crop:", avg_yield_crop.idxmax(), "| Best soil:", avg_yield_soil.idxmax())

# ============================================================
# PART C: Data Preparation
# ============================================================

# Q11. Feature Encoding
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print("\nQ11. Categorical columns:", categorical_cols)

df_clean = df.dropna().reset_index(drop=True)
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print("Q11. Shape after One-Hot Encoding:", df_encoded.shape)
print("Q11. First 5 rows:\n", df_encoded.head(5))

# Q12. Feature Selection
target_column = "Yield"
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]
print("\nQ12. Target variable:", target_column)
print("Q12. Input feature columns:", list(X.columns))

# ============================================================
# PART D: Machine Learning
# ============================================================

# Q13. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nQ13. X_train shape:", X_train.shape)
print("Q13. X_test shape :", X_test.shape)
print("Q13. y_train shape:", y_train.shape)
print("Q13. y_test shape :", y_test.shape)

# Q14. Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

coef_series = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nQ14. Intercept:", model.intercept_)
print("Q14. Coefficients:\n", coef_series)
print("Q14. Feature with highest positive coefficient:", coef_series.idxmax())
print("Q14. Test R^2 :", round(r2_score(y_test, y_pred), 4))
print("Q14. Test RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 4))