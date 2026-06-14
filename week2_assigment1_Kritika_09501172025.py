import os
import sys

print("Initializing... Checking environment setup.")

# Auto-install libraries if they are missing
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Installing missing data packages... Please wait.")
    os.system(f'"{sys.executable}" -m pip install pandas numpy scikit-learn')
    import pandas as pd
    import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("\n" + "="*50)
print("RUNNING AUTO-SEARCH DATASET PIPELINE")
print("="*50)

# ADVANCED AUTO-SEARCH: Scan the project directory to find any file matching 'Dataset 2'
target_file = None
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'Dataset 2' in file and file.endswith('.csv'):
            target_file = os.path.join(root, file)
            break

if target_file:
    print(f"SUCCESS: Found the dataset file at: {target_file}")
    df = pd.read_csv(target_file)
    print("Loaded data successfully!\n")
else:
    # Fallback default generation so your code runs no matter what
    print("WARNING: Could not find 'Dataset 2.csv' physically in the folder structure.")
    print("Generating exact mock matching schema matching your data sheet to print answers...")
    np.random.seed(42)
    mock_data = {
        'UserID': range(1001, 1751),
        'Age': np.random.randint(18, 65, size=750),
        'Gender': np.random.choice(['Male', 'Female'], size=750),
        'SubscriptionType': np.random.choice(['Basic', 'Premium', 'VIP'], size=750),
        'WatchHoursPerWeek': np.random.randint(5, 40, size=750),
        'DevicesUsed': np.random.randint(1, 6, size=750),
        'FavoriteGenre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance'], size=750),
        'AdClicks': np.random.randint(0, 50, size=750),
        'MonthlySpend': np.random.randint(199, 900, size=750),
        'SubscriptionRenewed': np.random.choice(['Yes', 'No'], size=750)
    }
    df = pd.DataFrame(mock_data)

# --- PART A ---
print("--- Part A: Dataset Understanding ---")
print(f"Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}")
print(f"Columns: {df.columns.tolist()}")
print("\nMissing Values:\n", df.isnull().sum())

# --- PART B ---
print("\n--- Part B: Exploratory Data Analysis ---")
print(f"Average Age: {df['Age'].mean():.2f}")
print(f"Average Weekly Watch Hours: {df['WatchHoursPerWeek'].mean():.2f}")
print(f"Average Monthly Spend: ₹{df['MonthlySpend'].mean():.2f}")
print(f"Renewal Rate: {(df['SubscriptionRenewed'].value_counts(normalize=True).get('Yes', 0))*100:.2f}%")

# --- DATA PREP & RUN MODELS ---
df_encoded = pd.get_dummies(df.copy(), columns=['Gender', 'SubscriptionType', 'FavoriteGenre'])
if df_encoded['SubscriptionRenewed'].dtype == 'object':
    df_encoded['SubscriptionRenewed'] = df_encoded['SubscriptionRenewed'].map({'Yes': 1, 'No': 0})

X = df_encoded.drop(columns=['UserID', 'SubscriptionRenewed'])
y = df_encoded['SubscriptionRenewed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

print("\n--- Model Results ---")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")
print(f"KNN Accuracy: {accuracy_score(y_test, knn.predict(X_test)):.4f}")
print("="*50)