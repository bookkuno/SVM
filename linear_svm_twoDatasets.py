import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Step 1: Load the Datasets
data1A = pd.read_csv("/Users/book_kuno/Downloads/DDoS 2018/02-20-2018.csv", low_memory=False)
data2A = pd.read_csv("/Users/book_kuno/Downloads/DDoS 2018/02-21-2018.csv", low_memory=False)

#-----------Customized part for each particular datasets--------------
# List of columns to drop
columns_to_drop = ['Flow ID', 'Src Port', 'Src IP', 'Dst IP']
# Drop the specified columns from data1
data1AD = data1A.drop(columns=columns_to_drop, errors='ignore')
# Randomly sample 1/10 of the data
data1 = data1AD.sample(frac=0.05, random_state=42)  # frac=0.1 means 10%, random_state ensures reproducibility
print(data1.head())
data2 = data2A.sample(frac=0.5, random_state=42)  # frac=0.1 means 10%, random_state ensures reproducibility
print(data2.head())
#-----------Customized part for each particular datasets--------------

# Step 2: Preprocess the Data
def preprocess_data(data):
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
    
    return X, y

X1, y1 = preprocess_data(data1)
X2, y2 = preprocess_data(data2)

# Step 3: Split the Training Data
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Step 4: Train the Linear SVM
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
#function is used to create an SVM classifier (from scikit-learn)
#svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Step 5: Validate the Model
y_val_pred = svm_model.predict(X_val)
print("Validation Results")
print(classification_report(y_val, y_val_pred))

# Step 6: Hyperparameter Tuning (Optional)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Step 7: Test the Model
y_test_pred = svm_model.predict(X2)
print("Test Results")
print(confusion_matrix(y2, y_test_pred))
print(classification_report(y2, y_test_pred))
print("Accuracy:", accuracy_score(y2, y_test_pred))