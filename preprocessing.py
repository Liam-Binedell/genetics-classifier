import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

features = pd.read_csv("data.csv", index_col=0)
labels = pd.read_csv("labels.csv", index_col=0).squeeze()

#print(features.shape) #(801, 20531) 
#print(labels.value_counts()) #distribution of the 5 cancer types 

#checking for missing values 
#print(features.isnull().sum().sum()) #is 0 so no cleaning needed :) 

#print(labels['Class'].value_counts()) #imbalanced class distribution...
#print(features.describe())


l_encoder = LabelEncoder()
labels_encoded = l_encoder.fit_transform(labels) #Lables to numbers 

#Train/Validation/Test split - 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels_encoded,
    test_size = 0.15,
    random_state = 42,
    stratify = labels_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size = 0.176, #15% of total
    random_state = 42,
    stratify = y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#removing features w/variance below threshold   
threshold = 0.5 #Could use 1.0 (?)
selector = VarianceThreshold(threshold)
X_train_filtered = selector.fit_transform(X_train)
X_val_filtered   = selector.transform(X_val)
X_test_filtered  = selector.transform(X_test)

print(f"Genes before filtering: {features.shape[1]}")
print(f"Genes after filtering: {X_train_filtered.shape[1]}")

#Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled   = scaler.transform(X_val_filtered)
X_test_scaled  = scaler.transform(X_test_filtered)

print(f"Final shapes — Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

#Testing thresholds...
# for t in [0.1, 0.5, 1.0, 1.5, 2.0]:
#     selector = VarianceThreshold(t)
#     filtered = selector.fit_transform(features)
#     print(f"Threshold {t:.1f}: {filtered.shape[1]:,} genes remaining")

np.save('X_train.npy', X_train_scaled)
np.save('X_val.npy',   X_val_scaled)
np.save('X_test.npy',  X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_val.npy',   y_val)
np.save('y_test.npy',  y_test)