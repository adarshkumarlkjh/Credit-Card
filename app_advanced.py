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
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please run the training script (`train_save_model.py`) first.")
        return None, None, None
    if not os.path.exists(ENCODERS_FILE):
        st.error(f"Error: Encoders file '{ENCODERS_FILE}' not found. Please run the training script (`train_save_model.py`) first.")
        return None, None, None
    if not os.path.exists(FEATURES_FILE):
        st.error(f"Error: Feature list file '{FEATURES_FILE}' not found. Please run the training script (`train_save_model.py`) first.")
        return None, None, None

    try:
        model = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        feature_columns = joblib.load(FEATURES_FILE)
        # Use st.success only once during loading, maybe after all are loaded.
        # Consider removing it if the app starts quickly enough.
        # st.success("Model, encoders, and feature list loaded successfully!")
        return model, encoders, feature_columns
    except Exception as e:
        st.error(f"An error occurred while loading prediction resources: {e}")
        return None, None, None

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("💳 Advanced Credit Card Fraud Detection")
# Use use_container_width instead of use_column_width
st.image("https://placehold.co/1200x300/E8F0FE/1967D2?text=Fraud+Detection+System&font=inter", use_container_width=True) # Placeholder banner
st.markdown("""
Welcome! This upgraded application uses a pre-trained **Logistic Regression** model to predict credit card fraud with improved speed and provides probability scores.

Fill in the transaction details in the sidebar to get a prediction.
""")

# Load the necessary files
model, encoders, feature_columns = load_prediction_resources()

# Only proceed if resources are loaded successfully
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
                        # Convert all options to string for consistency in selectbox
                        options_str = sorted([str(opt) for opt in encoders[col].classes_])
                        # Provide a default selection (e.g., the first option)
                        default_option = options_str[0] if options_str else ""

                        input_data[col] = target_col.selectbox(
                            label,
                            options=options_str,
                            # Find index robustly, handle if default_option isn't found
                            index=options_str.index(default_option) if default_option in options_str else 0,
                            key=widget_key
                        )
                    else:
                        target_col.warning(f"Could not load options for {label}. Please enter encoded value.")
                        input_data[col] = target_col.text_input(f"{label} (Encoded Value)", key=widget_key)
                except Exception as e:
                    target_col.error(f"Error creating dropdown for {label}: {e}")
                    input_data[col] = target_col.text_input(f"{label} (Enter Value)", key=widget_key) # Fallback

            # Explicitly check numeric types expected by the model
            elif col in ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'unix_time', 'merchant', 'category', 'gender', 'job']:
                 # Check if it was originally categorical (now encoded) or truly numeric
                 if col not in encoders: # Truly numeric features
                     default_val = 0.0
                     step = 0.01 if col in ['amt', 'lat', 'long', 'merch_lat', 'merch_long'] else 1.0
                     input_format = "%.2f" if step==0.01 else "%d" # Correct format string

                     # Define min/max if helpful, otherwise leave open or use reasonable defaults
                     min_val = None # Or np.finfo(np.float64).min
                     max_val = None # Or np.finfo(np.float64).max

                     input_data[col] = target_col.number_input(
                         label,
                         min_value=min_val,
                         max_value=max_val,
                         value=default_val,
                         step=step,
                         format=input_format,
                         key=widget_key
                     )
                 # else: Encoded categoricals are handled by the selectbox/text input above

            else: # Fallback for any unexpected column type (shouldn't happen if FEATURES_FILE is correct)
                input_data[col] = target_col.text_input(label, key=widget_key)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict Fraud Status')

    # --- Prediction Logic (runs when form is submitted) ---
    if submit_button:
        try:
            # Create DataFrame for prediction from form input
            predict_df = pd.DataFrame([input_data])
            # Create a copy for processing, keep original string inputs safe
            predict_df_processed = predict_df.copy()

            st.write("---")
            st.subheader("Processing Input...")

            # Apply encoding using loaded encoders
            encoding_errors = []
            cols_to_encode = [col for col in feature_columns if col in encoders]

            for col in cols_to_encode:
                value = input_data[col] # Get the value selected/entered by user
                try:
                    # Get the string representation of known classes
                    known_classes_str = [str(cls) for cls in encoders[col].classes_]
                    if str(value) in known_classes_str:
                        # Transform the user's selected string into its encoded integer
                        predict_df_processed[col] = encoders[col].transform([str(value)])[0]
                    else:
                        # Handle unseen labels robustly: e.g., assign a special value or raise error
                        encoding_errors.append(f"Category '{value}' for feature '{col.replace('_', ' ').title()}' was not seen during training. Cannot predict accurately.")
                        # Set a value that might cause issues later, forcing error handling
                        predict_df_processed[col] = np.nan # Or -1, depends on model robustness
                except Exception as e:
                     encoding_errors.append(f"Error encoding '{col.replace('_', ' ').title()}': {e}")


            if encoding_errors:
                 st.error("Input Errors Found:")
                 for err in encoding_errors:
                     st.error(f"- {err}")
                 st.warning("Prediction cannot proceed due to invalid categorical input.")

            else:
                 # Ensure correct column order
                 try:
                     predict_df_processed = predict_df_processed[feature_columns]
                 except KeyError as e:
                     st.error(f"Column Mismatch Error: Could not find expected column '{e}' in processed input.")
                     st.stop() # Stop execution if columns are wrong

                 # Convert columns to appropriate numeric types before prediction
                 conversion_errors = []
                 for col in feature_columns:
                     # Attempt conversion only if not already numeric (e.g., from text fallback)
                     if not pd.api.types.is_numeric_dtype(predict_df_processed[col]):
                         try:
                             # Use errors='coerce' to turn failures into NaN
                             predict_df_processed[col] = pd.to_numeric(predict_df_processed[col], errors='coerce')
                         except Exception as e:
                             # This catch might be redundant due to errors='coerce' but kept for safety
                             conversion_errors.append(f"Error converting '{col.replace('_', ' ').title()}' to numeric: {e}")

                     # Check for NaNs after coercion specifically for numeric columns
                     if col not in encoders and predict_df_processed[col].isnull().any():
                          conversion_errors.append(f"Invalid numeric input provided for '{col.replace('_', ' ').title()}'.")


                 if conversion_errors:
                     st.error("Data Type Errors Found:")
                     for err in conversion_errors:
                         st.error(f"- {err}")
                     st.warning("Prediction cannot proceed due to data type conversion errors.")

                 elif predict_df_processed.isnull().values.any():
                    # This check catches NaNs from encoding errors or failed numeric conversions
                    st.error("Error: Missing or invalid values detected after processing input. Please check all fields.")
                    st.dataframe(predict_df_processed.isnull().sum().to_frame(name='Missing/Invalid Count'))

                 else:
                     st.subheader("Making Prediction...")
                     # Ensure dtypes match the training data expected by the model (optional but good practice)
                     # Example: X_train.dtypes can be saved during training
                     # for col in predict_df_processed.columns:
                     #     predict_df_processed[col] = predict_df_processed[col].astype(X_train.dtypes[col])

                     # Make prediction
                     prediction = model.predict(predict_df_processed)
                     prediction_proba = model.predict_proba(predict_df_processed)

                     # --- Display Results ---
                     st.subheader("📊 Prediction Result:")
                     probability_fraud = prediction_proba[0][1] # Probability of class 1 (Fraud)

                     # Use columns for result layout
                     res_col1, res_col2 = st.columns([1, 4])

                     with res_col1:
                         if prediction[0] == 1:
                             st.image("https://placehold.co/100x100/FFCCCC/CC0000?text=ALERT&font=inter", width=100) # Placeholder Alert Icon
                         else:
                             st.image("https://placehold.co/100x100/CCFFCC/009900?text=OK&font=inter", width=100) # Placeholder OK Icon

                     with res_col2:
                         if prediction[0] == 1:
                             st.error(f"Transaction is **FRAUDULENT**")
                             st.metric(label="Fraud Confidence Score", value=f"{probability_fraud:.2%}")

                         else:
                             st.success(f"Transaction is **LEGITIMATE**")
                             st.metric(label="Fraud Probability Score", value=f"{probability_fraud:.2%}", delta=f"{-probability_fraud:.2%}", delta_color="inverse")


                     # Expander for input details
                     with st.expander("Show Input Details Provided"):
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
            # Display raw input data for debugging
            st.json({k: str(v) for k,v in input_data.items()}) # Convert to string for display

# This message shows if the .joblib files were not loaded successfully
else:
    st.warning("Required model files (`.joblib`) are missing or failed to load. Please ensure the `train_save_model.py` script has been run successfully in the same directory as this app.")

st.markdown("""
---
*Disclaimer: This prediction is based on a machine learning model and should be used for informational purposes only.*
""")

