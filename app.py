import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Optional: For evaluation during training
from sklearn.metrics import accuracy_score # Optional: For evaluation
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Data Loading ---
def load_data(file_path):
    """Loads and performs initial drop."""
    try:
        data = pd.read_csv(file_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

# --- Preprocessing ---
def preprocess_data_for_training(df):
    """Preprocesses the DataFrame for training."""
    print("Starting preprocessing...")
    # Drop unnecessary columns
    columns_to_drop = ['cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")

    # Handle missing values - Crucial step
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    rows_after_na = len(df_processed)
    print(f"Dropped {initial_rows - rows_after_na} rows due to missing values.")

    if df_processed.empty:
        print("Error: DataFrame is empty after dropping missing values.")
        return None, None, None

    # Convert target to integer
    if 'is_fraud' in df_processed.columns:
        df_processed['is_fraud'] = df_processed['is_fraud'].astype(int)

    # Separate features (X) and target (y)
    X = df_processed.drop(columns=['is_fraud'])
    y = df_processed['is_fraud']
    print("Separated features (X) and target (y).")

    # --- Encoding ---
    encoders = {}
    categorical_cols = ['merchant', 'category', 'gender', 'job']
    print("Encoding categorical features...")
    for col in categorical_cols:
        if col in X.columns:
            print(f"Encoding column: {col}")
            encoder = LabelEncoder()
            # Fit encoder and transform the column
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder # Store fitted encoder
        else:
             print(f"Warning: Categorical column '{col}' not found in features.")


    print("Preprocessing finished.")
    return X, y, encoders

# --- Main Training Logic ---
if __name__ == "__main__":
    DATA_FILE = 'fraudTrain.csv'
    MODEL_FILE = 'fraud_model.joblib'
    ENCODERS_FILE = 'fraud_encoders.joblib'
    FEATURES_FILE = 'fraud_features.joblib'

    # 1. Load Data
    data = load_data(DATA_FILE)

    if data is not None:
        # 2. Preprocess Data
        X, y, encoders = preprocess_data_for_training(data)

        if X is not None and y is not None:
            feature_columns = list(X.columns) # Get feature names *after* preprocessing and encoding

            # --- Optional: Split data for evaluation ---
            # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            # print("Data split for training and validation.")

            # 3. Train Model (Using Logistic Regression)
            print("Training Logistic Regression model...")
            model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
            model.fit(X, y) # Train on the full preprocessed data (X, y)
            # model.fit(X_train, y_train) # Or train only on the training split
            print("Model training complete.")

            # --- Optional: Evaluate Model ---
            # y_pred_val = model.predict(X_val)
            # accuracy = accuracy_score(y_val, y_pred_val)
            # print(f"Validation Accuracy: {accuracy:.4f}")

            # 4. Save Model, Encoders, and Feature List
            try:
                joblib.dump(model, MODEL_FILE)
                print(f"Model saved successfully to {MODEL_FILE}")
                joblib.dump(encoders, ENCODERS_FILE)
                print(f"Encoders saved successfully to {ENCODERS_FILE}")
                joblib.dump(feature_columns, FEATURES_FILE)
                print(f"Feature list saved successfully to {FEATURES_FILE}")
            except Exception as e:
                print(f"Error saving files: {e}")
        else:
            print("Could not proceed with training due to preprocessing issues.")
    else:
        print("Could not proceed with training because data loading failed.")
