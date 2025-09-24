import pandas as pd
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Save the processed data
print("Data preprocessing complete. Training and testing sets are ready.")