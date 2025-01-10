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
data1 = data1AD.sample(frac=0.01, random_state=42)  # frac=0.1 means 10%, random_state ensures reproducibility
print(data1.head())
data2 = data2A.sample(frac=0.1, random_state=42)  # frac=0.1 means 10%, random_state ensures reproducibility
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

X1, y1 = preprocess_data(data2)
X2, y2 = preprocess_data(data1)

# Step 3: Split the Training Data
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.3, random_state=42)

import numpy as np
from scipy.optimize import minimize

class CustomSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = None
    
    def rbf_kernel(self, X1, X2):
        """Radial Basis Function kernel."""
        return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)**2)
    
    def compute_kernel(self, X):
        """Compute the kernel matrix."""
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X, X)
        else:
            raise ValueError("Unsupported kernel type.")
    
    def fit(self, X, y):
        """Train the SVM model."""
        m, n = X.shape
        K = self.compute_kernel(X) * (y[:, np.newaxis] * y)
        
        # Define the dual optimization problem
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)
        
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
            {'type': 'ineq', 'fun': lambda alpha: self.C - alpha},
            {'type': 'ineq', 'fun': lambda alpha: alpha}
        ]
        
        # Initial alpha values
        alpha_init = np.zeros(m)
        
        # Solve the optimization problem
        result = minimize(objective, alpha_init, constraints=constraints)
        self.alpha = result.x
        
        # Extract support vectors
        support_indices = self.alpha > 1e-5
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]
        self.alpha = self.alpha[support_indices]
        
        # Calculate bias
        self.bias = np.mean(
            self.support_labels - np.sum(self.alpha * self.support_labels * K[support_indices], axis=1)
        )
    
    def predict(self, X):
        """Predict using the trained SVM."""
        K = self.rbf_kernel(X, self.support_vectors)
        predictions = np.sum(K * (self.alpha * self.support_labels), axis=1) + self.bias
        return np.sign(predictions)

# Usage of CustomSVM
custom_svm = CustomSVM(C=1.0, kernel='rbf', gamma=0.1)
custom_svm.fit(X_train.values, y_train.values)
y_val_pred_custom = custom_svm.predict(X_val.values)

# Validate
from sklearn.metrics import classification_report
print("Validation Results (Custom SVM):")
print(classification_report(y_val, y_val_pred_custom))

# Custom Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

best_params = None
best_score = -np.inf

for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        custom_svm = CustomSVM(C=C, kernel='rbf', gamma=gamma)
        
        # Perform 3-fold cross-validation
        scores = []
        for train_idx, val_idx in KFold(n_splits=3, shuffle=True, random_state=42).split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train and validate
            custom_svm.fit(X_train_cv.values, y_train_cv.values)
            y_val_pred = custom_svm.predict(X_val_cv.values)
            scores.append(accuracy_score(y_val_cv, y_val_pred))
        
        # Average validation score
        mean_score = np.mean(scores)
        print(f"C={C}, gamma={gamma}, mean_accuracy={mean_score:.4f}")
        
        # Track the best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'C': C, 'gamma': gamma}

print("Best Parameters:", best_params)

# Train the model with the best hyperparameters
final_svm = CustomSVM(C=best_params['C'], kernel='rbf', gamma=best_params['gamma'])
final_svm.fit(X_train.values, y_train.values)

# Predict on the test set
y_test_pred_custom = final_svm.predict(X2.values)

# Evaluate the model on the test set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Test Results (Custom SVM):")
print(confusion_matrix(y2, y_test_pred_custom))
print(classification_report(y2, y_test_pred_custom))
print("Accuracy:", accuracy_score(y2, y_test_pred_custom))
