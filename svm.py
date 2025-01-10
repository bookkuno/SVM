import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Step 2: Load the Dataset
# data = pd.read_csv("/Users/book_kuno/Downloads/archive/02-14-2018.csv")
# print(data.head())
data = pd.read_csv("/Users/book_kuno/Downloads/DDoS 2018/02-21-2018.csv")
print(data.head())
# Randomly sample 1/10 of the data
data = data.sample(frac=0.1, random_state=42)  # frac=0.1 means 10%, random_state ensures reproducibility
# Preview the sampled data
print(data.head())

# Step 3: Preprocess the Data
data.columns = data.columns.str.strip()
data = data.dropna()
# Encode the target column ('Label')
encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])
# Select only numeric columns for scaling
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('Label')  # Exclude the target column
# Check for infinite or extremely large values
data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=numeric_columns)
# Scale the numeric feature columns
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
X = data[numeric_columns]
y = data['Label']

# Step 4: Split the Dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 5: Train the Linear SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Step 6: Validate the Model
y_val_pred = svm_model.predict(X_val)
print("Validation Results")
print(classification_report(y_val, y_val_pred))

# Step 7: Hyperparameter Tuning (Optional)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Step 8: Test the Model
y_test_pred = svm_model.predict(X_test)
print("Test Results")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Accuracy:", accuracy_score(y_test, y_test_pred))