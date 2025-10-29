import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os # To check if files exist

warnings.filterwarnings('ignore')

# --- File Paths ---
MODEL_FILE = 'fraud_model.joblib'
ENCODERS_FILE = 'fraud_encoders.joblib'
FEATURES_FILE = 'fraud_features.joblib'

# --- Load Pre-trained Model, Encoders, and Features ---
# Cache these resources for performance
@st.cache_resource
def load_prediction_resources():
    """Loads the pre-trained model, encoders, and feature list."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please run the training script first.")
        return None, None, None
    if not os.path.exists(ENCODERS_FILE):
        st.error(f"Error: Encoders file '{ENCODERS_FILE}' not found. Please run the training script first.")
        return None, None, None
    if not os.path.exists(FEATURES_FILE):
        st.error(f"Error: Feature list file '{FEATURES_FILE}' not found. Please run the training script first.")
        return None, None, None

    try:
        model = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        feature_columns = joblib.load(FEATURES_FILE)
        st.success("Model, encoders, and feature list loaded successfully!")
        return model, encoders, feature_columns
    except Exception as e:
        st.error(f"An error occurred while loading prediction resources: {e}")
        return None, None, None

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("💳 Advanced Credit Card Fraud Detection")
st.image("https://placehold.co/1200x300/E8F0FE/1967D2?text=Fraud+Detection+System&font=inter", use_column_width=True) # Placeholder banner
st.markdown("""
Welcome! This upgraded application uses a pre-trained **Logistic Regression** model to predict credit card fraud with improved speed and provides probability scores.

Fill in the transaction details in the sidebar to get a prediction.
""")

# Load the necessary files
model, encoders, feature_columns = load_prediction_resources()

if model and encoders and feature_columns:
    st.sidebar.header("Enter Transaction Details:")

    # --- User Input Form ---
    with st.sidebar.form(key='transaction_form'):
        input_data = {}
        widget_key_counter = 0

        # Use columns for better layout in the sidebar
        col1, col2 = st.columns(2)

        # Dynamically create input fields based on loaded feature columns
        for i, col in enumerate(feature_columns):
            target_col = col1 if i % 2 == 0 else col2 # Alternate columns
            widget_key = f"input_{col}_{widget_key_counter}" # Unique key for each widget
            widget_key_counter += 1
            label = col.replace('_', ' ').title()

            if col in encoders: # Categorical features - use selectbox
                try:
                    # Get original categories from the loaded encoder
                    if hasattr(encoders[col], 'classes_'):
                        options = list(encoders[col].classes_)
                        # Ensure options are strings for selectbox, sort them
                        options_str = sorted([str(opt) for opt in options])
                        # Provide a default selection (e.g., the first option)
                        default_option = options_str[0] if options_str else ""

                        input_data[col] = target_col.selectbox(
                            label,
                            options=options_str,
                            index=options_str.index(default_option) if default_option in options_str else 0,
                            key=widget_key
                        )
                    else:
                        target_col.warning(f"Could not load options for {label}. Please enter encoded value.")
                        input_data[col] = target_col.text_input(f"{label} (Encoded Value)", key=widget_key)
                except Exception as e:
                    target_col.error(f"Error creating dropdown for {label}: {e}")
                    input_data[col] = target_col.text_input(f"{label} (Enter Value)", key=widget_key) # Fallback

            elif col in ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'unix_time']: # Known numeric features
                 # Use number_input with reasonable defaults/steps
                 # For simplicity, we won't fetch min/max here, but you could save them during training too
                 default_val = 0.0
                 step = 0.01 if col in ['amt', 'lat', 'long', 'merch_lat', 'merch_long'] else 1.0
                 input_data[col] = target_col.number_input(
                     label,
                     value=default_val,
                     step=step,
                     format="%f" if step==0.01 else "%d", # Format based on step
                     key=widget_key
                 )
            else: # Fallback for any other type (treat as text initially)
                input_data[col] = target_col.text_input(label, key=widget_key)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict Fraud Status')

    # --- Prediction Logic (runs when form is submitted) ---
    if submit_button:
        try:
            # Create DataFrame for prediction from form input
            predict_df = pd.DataFrame([input_data])
            predict_df_processed = predict_df.copy() # Keep original for display if needed

            st.write("---")
            st.subheader("Processing Input...")

            # Apply encoding using loaded encoders
            encoding_errors = []
            for col, value in input_data.items():
                if col in encoders:
                    try:
                        # Transform the user's selected string into its encoded integer
                        # Handle potential unseen labels during prediction
                        if value in encoders[col].classes_:
                            predict_df_processed[col] = encoders[col].transform([value])[0]
                        else:
                            # Option 1: Assign a default/unknown category encoding (e.g., -1 or len(classes))
                            # predict_df_processed[col] = -1 # Or another placeholder
                            # Option 2: Raise an error or warning
                            encoding_errors.append(f"Category '{value}' for feature '{col}' was not seen during training.")
                            # For now, let's cause an error later if it remains string
                    except Exception as e:
                         encoding_errors.append(f"Error encoding '{col}': {e}")


            if encoding_errors:
                 for err in encoding_errors:
                     st.error(err)
                 st.warning("Prediction cannot proceed due to encoding errors.")

            else:
                 # Ensure correct column order and dtypes
                 predict_df_processed = predict_df_processed[feature_columns] # Reorder columns

                 # Convert columns to appropriate numeric types before prediction
                 conversion_errors = []
                 for col in feature_columns:
                     # Check if column should be numeric (all except known categorical handled by encoder)
                     # This check assumes all features used in training were either encoded or numeric
                     if col not in encoders:
                         try:
                             predict_df_processed[col] = pd.to_numeric(predict_df_processed[col])
                         except ValueError:
                             conversion_errors.append(f"Could not convert input for '{col}' to a number.")
                         except Exception as e:
                             conversion_errors.append(f"Error converting '{col}' to numeric: {e}")

                 if conversion_errors:
                     for err in conversion_errors:
                         st.error(err)
                     st.warning("Prediction cannot proceed due to data type conversion errors.")

                 elif predict_df_processed.isnull().values.any():
                    st.error("Error: Missing values detected after processing input. Please check all fields.")
                    st.dataframe(predict_df_processed.isnull().sum().to_frame(name='Missing Count'))

                 else:
                     st.subheader("Making Prediction...")
                     # Make prediction
                     prediction = model.predict(predict_df_processed)
                     prediction_proba = model.predict_proba(predict_df_processed)

                     # --- Display Results ---
                     st.subheader("📊 Prediction Result:")
                     probability_fraud = prediction_proba[0][1] # Probability of class 1 (Fraud)

                     if prediction[0] == 1:
                         st.error(f"🚨 Transaction is **FRAUDULENT** (Confidence: {probability_fraud:.2%})")
                     else:
                         st.success(f"✅ Transaction is **LEGITIMATE** (Fraud Probability: {probability_fraud:.2%})")

                     # Expander for details
                     with st.expander("Show Input Details"):
                         # Display user input for confirmation (original string values)
                         display_input = {}
                         for k, v in input_data.items():
                             display_input[k.replace('_', ' ').title()] = v
                         st.json(display_input)


        except KeyError as e:
            st.error(f"Error during prediction: Feature mismatch - {e}. The model expected a feature that wasn't provided or processed correctly.")
            st.info(f"Model expects features: {feature_columns}")
        except ValueError as e:
             st.error(f"Error during prediction processing: {e}. Check if all inputs have the correct format (e.g., numbers).")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.error("Input data at time of error:")
            st.write(input_data) # Log input data for debugging

else:
    st.warning("Required model files (`.joblib`) are missing. Please run the `train_save_model.py` script first in the same directory.")

st.markdown("""
---
*Disclaimer: This prediction is based on a machine learning model and should be used for informational purposes only.*
""")
