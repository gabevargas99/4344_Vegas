import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Function to load data


def load_data(filepath):
    return pd.read_csv(filepath, delimiter=';')

# Function to inspect the data


def inspect_data(df):
    print(df.info())
    print(df.describe())
    print(df.head())

# Univariate Analysis
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()


def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Bivariate Analysis
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f'Scatter plot of {x_col} vs {y_col}')
    plt.show()


def plot_grouped_bar(df, x_col, y_col):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=x_col, y=y_col, data=df, ci=None)
    plt.title(f'Average {y_col} by {x_col}')
    plt.xticks(rotation=45)
    plt.show()

# Multivariate Analysis
def plot_pairplot(df, columns):
    sns.pairplot(df[columns])
    plt.show()

# Function to encode categorical columns
def label_encode_columns(df, columns):
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:  # Check if the column exists in the DataFrame
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            print(f"Column not found: {col}")
    return df

# Function to plot the correlation matrix
def plot_correlation_matrix(df):
    # Optionally encode or select numeric columns here
    # Ensure df is numeric if not pre-processed
    df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def traveler_type_classification_1(df):
    # Ensure that the columns are in the correct format (numeric)
    categorical_features = ['Pool', 'Gym', 'Tennis court',
                            'Spa', 'Casino', 'Free internet', 'Period of stay']
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Selecting features and target
    X = df[['Score', 'Pool', 'Gym', 'Tennis court', 'Spa',
            'Casino', 'Free internet', 'Period of stay']]
    y = df['Traveler type']
    print(df)
    # Encode the target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Create a classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    classifier.fit(X_train, y_train)

    # Predicting the test results
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def clean_stars(df, stars_column='Hotel stars'):
    df = df.dropna().reset_index(drop=True)
    # Replace commas with periods to handle European decimal notation
    df[stars_column] = df[stars_column].replace(',', '.', regex=True)
    # Convert the 'Hotel stars' column to float, coercing errors to NaN
    df[stars_column] = pd.to_numeric(df[stars_column], errors='coerce')
    # Round the 'Hotel stars' column to the nearest whole number
    df[stars_column] = df[stars_column].round().astype('Int64')  # Use 'Int64' (capital "I") to allow NaN handling
    # Remove rows with NaN values across the entire DataFrame
    
    return df

def get_traveller_type_details(df):
    class_counts = df['Traveler type'].value_counts()
    print(class_counts)

def rebalance_traveler_type(X, y):
    X_encoded = X.apply(LabelEncoder().fit_transform)
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

def traveler_type_classification(df):
    df_encoded = label_encode_columns(df, ['User country', 'Traveler type', 'Score', 'Pool', 'Gym', 'Tennis court', 'Spa', 'Casino', 'Free internet', 'Period of stay', 'Hotel name', 'User continent', 'Review month', 'Review weekday'])
    X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop('Traveler type', axis=1), df_encoded['Traveler type'], test_size=0.3, random_state=42, stratify=df_encoded['Traveler type'])
    X_train_resampled, y_train_resampled = rebalance_traveler_type(X_train, y_train)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_resampled, y_train_resampled)
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))  # Handling zero division explicitly

# Example usage
if __name__ == '__main__':
    df = load_data('LasVegasTripAdvisorReviews-Dataset.csv')
    inspect_data(df)
    plot_histogram(df, 'Score')
    plot_boxplot(df, 'Helpful votes')
    plot_scatter(df, 'Helpful votes', 'Score')
    plot_grouped_bar(df, 'Traveler type', 'Score')
    plot_pairplot(df, ['Score', 'Nr. reviews', 'Helpful votes'])
    df = clean_stars(df)
    categorical_columns = ['User country', 'Traveler type', 'Score', 'Pool', 'Gym', 'Tennis court', 'Spa',
                           'Casino', 'Free internet', 'Period of stay', 'Hotel name', 'User continent', 'Review month', 'Review weekday']
    df_encoded = label_encode_columns(df, categorical_columns)
    get_traveller_type_details(df)
    traveler_type_classification(df)
    plot_correlation_matrix(df_encoded)
    
