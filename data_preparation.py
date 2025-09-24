import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('enhanced_household_power_usage.csv')

# Check for missing values
missing_values = data.isnull().sum()

# Check data types
data_types = data.dtypes

# Print results
print("Missing Values:\n", missing_values)
print("\nData Types:\n", data_types)

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Define features and target
X = data.drop(columns=['Power_Usage_kWh'])
y = data['Power_Usage_kWh']

# Define preprocessing for numerical and categorical columns
numerical_features = ['Hour', 'Num_Occupants']
categorical_features = ['Appliance', 'Weather', 'Season']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Preprocess the data
X_preprocessed = pipeline.fit_transform(X)

# Temporarily create DataFrame without column names to inspect structure
X_preprocessed = pd.DataFrame(X_preprocessed)
print("Shape of X_preprocessed:", X_preprocessed.shape)
print("First few rows of X_preprocessed:")
print(X_preprocessed.head())

# Dynamically generate column names for one-hot encoded features
categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(categorical_feature_names)

# Assign correct column names to X_preprocessed
X_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)

# Perform advanced feature engineering
X_preprocessed['hour_sin'] = np.sin(2 * np.pi * X_preprocessed['Hour'] / 24)
X_preprocessed['hour_cos'] = np.cos(2 * np.pi * X_preprocessed['Hour'] / 24)
X_preprocessed['Occupants_Usage'] = X_preprocessed['Num_Occupants'] * y.values

# Inspect the structure of X_preprocessed
print("Shape of X_preprocessed:", X_preprocessed.shape)
print("First few rows of X_preprocessed:")
print(X_preprocessed.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Save the preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Data preprocessing complete. Training and testing sets are ready.")
print("Preprocessed data saved as .npy files.")