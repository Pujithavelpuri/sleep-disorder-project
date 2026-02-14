# app.py - Complete Streamlit App with ML Integration using Excel file

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import sys

# Check for openpyxl and provide installation instructions if missing
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="centered"
)

# Title and description
st.title("😴 Sleep Disorder Prediction System")
st.markdown(
    """
    This application predicts the type of sleep disorder (None, Insomnia, Sleep Apnea)  
    based on health and lifestyle indicators. Fill in the patient details below to get a prediction.
    """
)

# Check for openpyxl and show installation instructions if missing
if not OPENPYXL_AVAILABLE:
    st.error("⚠️ Missing required dependency: openpyxl")
    st.markdown("""
    ### Installation Instructions:
    
    **Option 1: Install via pip (recommended)**
    ```bash
    pip install openpyxl
    ```
    
    **Option 2: If using Streamlit Cloud**
    Add this to your `requirements.txt` file:
    ```
    openpyxl
    ```
    
    After installing, please restart the app.
    """)
    
    # Option to install directly from the app
    if st.button("🔄 Attempt to install openpyxl now"):
        with st.spinner("Installing openpyxl..."):
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
                st.success("✅ openpyxl installed successfully! Please restart the app.")
                st.info("Click the 'Rerun' button in the top-right corner to restart.")
            except Exception as e:
                st.error(f"Installation failed: {str(e)}")
    
    st.stop()

# Function to load dataset from Excel file
@st.cache_resource
def load_excel_dataset():
    """Load the sleep disorder dataset from Excel file"""
    excel_file = "sleep_disorder_powerbi.xlsx"
    
    # Check if file exists
    if os.path.exists(excel_file):
        try:
            # Try reading with openpyxl engine
            df = pd.read_excel(excel_file, engine='openpyxl')
            
            
            # Clean the dataset
            # 1. Replace NaN in Sleep Disorder with 'None'
            df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
            
            # 2. Reset index
            df = df.reset_index(drop=True)
            
            # 3. Split blood pressure if needed
            if 'Blood Pressure' in df.columns and 'Systolic_BP' not in df.columns:
                try:
                    # Handle potential formatting issues
                    bp_split = df['Blood Pressure'].str.split("/", expand=True)
                    df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
                    df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
                    
                    # Fill any NaN values with median
                    df['Systolic_BP'] = df['Systolic_BP'].fillna(df['Systolic_BP'].median())
                    df['Diastolic_BP'] = df['Diastolic_BP'].fillna(df['Diastolic_BP'].median())
                    
                    
                except Exception as e:
                    
            
            # 4. Drop Person ID if it exists (not needed for prediction)
            if 'Person ID' in df.columns:
                df = df.drop(columns=['Person ID'])
            
            # 5. Fill any remaining NaN values in numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # 6. Fill any remaining NaN values in categorical columns with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            # Display basic info about the dataset
            
            
            # Save as CSV for backup and faster loading next time
            df.to_csv('sleep_cleaned.csv', index=False)
            
            
            return df
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return None
    else:
        st.error(f"❌ Excel file '{excel_file}' not found in root directory!")
        return None

# Function to train the model with debugging
@st.cache_resource
def train_model():
    """Train the Random Forest model on the dataset with debugging"""
    
    # Load from Excel
    df = load_excel_dataset()
    
    if df is None:
        st.error("Failed to load dataset")
        return None, None, None, None, None, None, None, None
    
    # Display dataset info in main area
    st.success("✅ Dataset loaded successfully!")
    
    with st.expander("📊 Dataset Overview", expanded=True):
        st.write("**First few rows:**")
        st.dataframe(df.head())
        
        # Show distribution of target variable
        st.write("**Target Variable Distribution:**")
        target_dist = df['Sleep Disorder'].value_counts()
        st.dataframe(target_dist)
        
        # Create a bar chart of the distribution
        st.bar_chart(target_dist)
        
        # Show data types
        st.write("**Data Types:**")
        st.dataframe(df.dtypes)
        
        # Show basic statistics
        st.write("**Basic Statistics:**")
        st.dataframe(df.describe())
    
    # Prepare features and target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    st.write(f"**Categorical features:** {categorical_cols}")
    st.write(f"**Numerical features:** {numerical_cols}")
    
    # Encode categorical variables
    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle any NaN values
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le
        st.write(f"Encoded {col} with {len(le.classes_)} classes: {le.classes_.tolist()}")
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    st.write(f"**Target classes:** {target_encoder.classes_.tolist()}")
    
    # Create a mapping of encoded values to class names
    class_mapping = dict(zip(range(len(target_encoder.classes_)), target_encoder.classes_))
    st.write(f"**Class mapping:** {class_mapping}")
    
    # Count samples per class
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_counts = dict(zip([class_mapping[u] for u in unique], counts))
    st.write(f"**Target distribution after encoding:** {class_counts}")
    
    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Calculate class weights manually for better balance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
    st.write(f"**Class weights:** {class_weight_dict}")
    
    # Train model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    st.write(f"**Training set size:** {len(X_train)}")
    st.write(f"**Test set size:** {len(X_test)}")
    
    # Train the model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
    
    # Calculate and store accuracy for display
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    st.write(f"**Training accuracy:** {train_score:.2%}")
    st.write(f"**Testing accuracy:** {test_score:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.write("**Top 5 important features:**")
    st.dataframe(feature_importance.head(5))
    
    # Test predictions on sample data
    st.write("**Sample predictions on test data (first 10 samples):**")
    
    # Get predictions for test set
    y_pred = model.predict(X_test)
    
    # Convert back to original labels
    y_test_labels = target_encoder.inverse_transform(y_test)
    y_pred_labels = target_encoder.inverse_transform(y_pred)
    
    # Create a DataFrame with actual vs predicted for first 10 samples
    results_df = pd.DataFrame({
        'Actual': y_test_labels[:10],
        'Predicted': y_pred_labels[:10]
    })
    st.dataframe(results_df)
    
    # Calculate and display accuracy per class
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("**Classification Report:**")
    st.dataframe(report_df)
    
    # Save model and encoders
    try:
        joblib.dump(model, 'best_sleep_model.pkl')
        joblib.dump(feature_encoders, 'feature_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(numerical_cols, 'numerical_cols.pkl')
        st.sidebar.success("✅ Model saved successfully!")
    except Exception as e:
        st.sidebar.warning(f"Could not save model files: {str(e)}")
    
    return model, feature_encoders, target_encoder, feature_names, train_score, test_score, feature_importance, scaler, numerical_cols, class_mapping

# Main app
def main():
    # Load or train the model
    with st.spinner("Loading AI model and training on Excel data..."):
        model_data = train_model()
        
        if model_data[0] is not None:
            model, feature_encoders, target_encoder, feature_names, train_score, test_score, feature_importance, scaler, numerical_cols, class_mapping = model_data
            
            # Sidebar content
            with st.sidebar:
                st.header("📈 Model Performance")
                st.metric("Training Accuracy", f"{train_score:.2%}")
                st.metric("Testing Accuracy", f"{test_score:.2%}")
                st.metric("Number of Classes", len(target_encoder.classes_))
                
                # Show class names
                st.write("**Classes:**")
                for cls in target_encoder.classes_:
                    st.write(f"  - {cls}")
                
                st.divider()
                
                # Feature importance
                st.header("🔝 Top Features")
                for i, row in feature_importance.head(5).iterrows():
                    st.write(f"{i+1}. **{row['feature']}**: {row['importance']:.3f}")
            
            # Create input form
            st.header("📋 Enter Patient Details")
            
            # Get feature names from dataset to create dynamic form
            categorical_features = [col for col in feature_encoders.keys()]
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            input_dict = {}
            
            with col1:
                # Age input
                if 'Age' in feature_names:
                    input_dict['Age'] = st.number_input("Age", min_value=1, max_value=100, value=45, step=1)
                
                # Gender input
                if 'Gender' in categorical_features:
                    gender_options = feature_encoders['Gender'].classes_.tolist() if 'Gender' in feature_encoders else ["Male", "Female"]
                    input_dict['Gender'] = st.selectbox("Gender", gender_options)
                
                # Occupation input
                if 'Occupation' in categorical_features:
                    occupation_options = feature_encoders['Occupation'].classes_.tolist() if 'Occupation' in feature_encoders else []
                    input_dict['Occupation'] = st.selectbox("Occupation", occupation_options, index=0)
                
                # Sleep Duration
                if 'Sleep Duration' in feature_names:
                    input_dict['Sleep Duration'] = st.slider("Sleep Duration (hours)", 3.0, 10.0, 6.5, step=0.1)
                
                # Quality of Sleep
                if 'Quality of Sleep' in feature_names:
                    input_dict['Quality of Sleep'] = st.slider("Quality of Sleep (1-10)", 1, 10, 6, step=1)
                
                # Physical Activity Level
                if 'Physical Activity Level' in feature_names:
                    input_dict['Physical Activity Level'] = st.slider("Physical Activity Level (minutes/day)", 0, 120, 50, step=5)
            
            with col2:
                # Stress Level
                if 'Stress Level' in feature_names:
                    input_dict['Stress Level'] = st.slider("Stress Level (1-10)", 1, 10, 6, step=1)
                
                # BMI Category
                if 'BMI Category' in categorical_features:
                    bmi_options = feature_encoders['BMI Category'].classes_.tolist() if 'BMI Category' in feature_encoders else ["Normal", "Overweight", "Obese"]
                    # Filter out any None values
                    bmi_options = [opt for opt in bmi_options if opt is not None and str(opt) != 'nan']
                    input_dict['BMI Category'] = st.selectbox("BMI Category", bmi_options, index=0)
                
                # Blood Pressure input
                if 'Systolic_BP' in feature_names and 'Diastolic_BP' in feature_names:
                    st.markdown("**Blood Pressure**")
                    bp_col1, bp_col2 = st.columns(2)
                    with bp_col1:
                        systolic = st.number_input("Systolic", min_value=80, max_value=200, value=130)
                    with bp_col2:
                        diastolic = st.number_input("Diastolic", min_value=50, max_value=130, value=85)
                    
                    # Store both the combined and split values
                    input_dict['Blood Pressure'] = f"{systolic}/{diastolic}"
                    input_dict['Systolic_BP'] = systolic
                    input_dict['Diastolic_BP'] = diastolic
                
                # Heart Rate
                if 'Heart Rate' in feature_names:
                    input_dict['Heart Rate'] = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=75)
                
                # Daily Steps
                if 'Daily Steps' in feature_names:
                    input_dict['Daily Steps'] = st.number_input("Daily Steps", min_value=0, max_value=20000, value=6000, step=100)
            
            # Add test buttons for different scenarios
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧪 Test: Healthy Profile"):
                    # Set values for a healthy person (should predict None)
                    st.session_state['test_input'] = {
                        'Age': 30,
                        'Gender': 'Female',
                        'Occupation': 'Engineer',
                        'Sleep Duration': 8.0,
                        'Quality of Sleep': 9,
                        'Physical Activity Level': 80,
                        'Stress Level': 3,
                        'BMI Category': 'Normal',
                        'Blood Pressure': '118/75',
                        'Systolic_BP': 118,
                        'Diastolic_BP': 75,
                        'Heart Rate': 68,
                        'Daily Steps': 9000
                    }
                    st.rerun()
            
            with col2:
                if st.button("🧪 Test: Insomnia Profile"):
                    # Set values for insomnia
                    st.session_state['test_input'] = {
                        'Age': 35,
                        'Gender': 'Female',
                        'Occupation': 'Teacher',
                        'Sleep Duration': 5.0,
                        'Quality of Sleep': 3,
                        'Physical Activity Level': 30,
                        'Stress Level': 8,
                        'BMI Category': 'Normal',
                        'Blood Pressure': '125/82',
                        'Systolic_BP': 125,
                        'Diastolic_BP': 82,
                        'Heart Rate': 78,
                        'Daily Steps': 4000
                    }
                    st.rerun()
            
            with col3:
                if st.button("🧪 Test: Sleep Apnea Profile"):
                    # Set values for sleep apnea
                    st.session_state['test_input'] = {
                        'Age': 55,
                        'Gender': 'Male',
                        'Occupation': 'Manager',
                        'Sleep Duration': 6.0,
                        'Quality of Sleep': 4,
                        'Physical Activity Level': 35,
                        'Stress Level': 7,
                        'BMI Category': 'Obese',
                        'Blood Pressure': '140/90',
                        'Systolic_BP': 140,
                        'Diastolic_BP': 90,
                        'Heart Rate': 82,
                        'Daily Steps': 4500
                    }
                    st.rerun()
            
            # Check if there's a test input in session state
            if 'test_input' in st.session_state:
                input_dict = st.session_state['test_input']
                st.info("Using test profile. You can modify the values above.")
            
            # Prediction button
            if st.button("🔍 Predict Sleep Disorder", type="primary"):
                
                # Create DataFrame
                input_data = pd.DataFrame([input_dict])
                
                st.write("**Input data for prediction:**")
                st.dataframe(input_data)
                
                try:
                    # Make a copy for encoding
                    input_encoded = input_data.copy()
                    
                    # Encode categorical variables using saved encoders
                    for col in feature_encoders.keys():
                        if col in input_encoded.columns:
                            # Handle unknown categories
                            try:
                                val = str(input_encoded[col].values[0])
                                # Check if value is in encoder classes
                                if val in feature_encoders[col].classes_:
                                    input_encoded[col] = feature_encoders[col].transform([val])[0]
                                    st.write(f"Encoded {col}: {input_dict[col]} → {input_encoded[col].values[0]}")
                                else:
                                    st.warning(f"Unknown {col} value '{val}'. Using most common class.")
                                    # Use the most frequent class (0)
                                    input_encoded[col] = 0
                            except Exception as e:
                                st.warning(f"Error encoding {col}: {str(e)}. Using default.")
                                input_encoded[col] = 0
                    
                    # Make sure all feature columns are present and in correct order
                    for col in feature_names:
                        if col not in input_encoded.columns:
                            # For missing numerical columns, use median from training
                            if col in numerical_cols:
                                input_encoded[col] = 0  # Will be scaled
                            else:
                                input_encoded[col] = 0
                            st.write(f"Added missing column {col} with default value 0")
                    
                    # Reorder columns to match feature_names
                    input_encoded = input_encoded[feature_names]
                    
                    # Scale numerical features
                    if numerical_cols and scaler is not None:
                        # Get the numerical columns that are in the input
                        cols_to_scale = [col for col in numerical_cols if col in input_encoded.columns]
                        if cols_to_scale:
                            input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])
                    
                    # Make prediction
                    prediction_encoded = model.predict(input_encoded)[0]
                    prediction_proba = model.predict_proba(input_encoded)[0]
                    
                    # Decode prediction
                    prediction = target_encoder.inverse_transform([prediction_encoded])[0]
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Define colors and icons for each prediction
                    if prediction == "None":
                        bg_color = "#28a745"
                        icon = "✅"
                        description = "No sleep disorder detected. Your sleep patterns appear normal."
                    elif prediction == "Insomnia":
                        bg_color = "#ffc107"
                        icon = "🌙"
                        description = "Insomnia detected. You may have difficulty falling or staying asleep."
                    elif prediction == "Sleep Apnea":
                        bg_color = "#dc3545"
                        icon = "⚠️"
                        description = "Sleep Apnea detected. Your breathing may be interrupted during sleep."
                    else:
                        bg_color = "#6c757d"
                        icon = "❓"
                        description = f"Prediction: {prediction}"
                    
                    # Display prediction with styling
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {bg_color};
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                            margin: 20px 0;
                        ">
                            <h1 style="font-size: 48px;">{icon}</h1>
                            <h2>Prediction: {prediction}</h2>
                            <p style="font-size: 16px;">{description}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display probability scores
                    st.subheader("📊 Probability Scores")
                    
                    # Get class labels
                    class_labels = target_encoder.classes_
                    
                    # Create probability DataFrame
                    proba_df = pd.DataFrame({
                        'Sleep Disorder': class_labels,
                        'Probability': prediction_proba * 100
                    })
                    
                    # Display as bar chart
                    st.bar_chart(proba_df.set_index('Sleep Disorder'))
                    
                    # Display detailed probabilities
                    for label, prob in zip(class_labels, prediction_proba):
                        prob_percent = prob * 100
                        if label == prediction:
                            st.markdown(f"**{label}:** {prob_percent:.1f}% ⭐")
                        else:
                            st.markdown(f"**{label}:** {prob_percent:.1f}%")
                    
                    # Additional health recommendations
                    st.subheader("💡 Health Recommendations")
                    
                    if prediction == "None":
                        st.info("""
                        ✅ **Maintain your current healthy habits:**
                        - Stick to your consistent sleep schedule
                        - Continue regular physical activity
                        - Keep stress levels low with relaxation techniques
                        - Monitor your sleep quality periodically
                        """)
                    elif prediction == "Insomnia":
                        st.warning("""
                        🌙 **Tips to improve sleep:**
                        - Establish a consistent sleep schedule (same bedtime/wake time)
                        - Create a relaxing bedtime routine (reading, gentle stretching)
                        - Avoid caffeine, alcohol, and screens 2-3 hours before bed
                        - Keep your bedroom cool, dark, and quiet
                        - Consider cognitive behavioral therapy for insomnia (CBT-I)
                        - Consult a sleep specialist if symptoms persist
                        """)
                    elif prediction == "Sleep Apnea":
                        st.error("""
                        ⚠️ **Important steps to take:**
                        - **Consult a healthcare provider immediately** for proper diagnosis
                        - Ask about a sleep study (polysomnography)
                        - Consider using a CPAP machine if prescribed
                        - Maintain a healthy weight through diet and exercise
                        - Sleep on your side instead of your back
                        - Avoid alcohol and sedatives before bedtime
                        - Elevate the head of your bed
                        - Quit smoking if applicable
                        """)
                    else:
                        st.info(f"Prediction: {prediction}. Please consult a healthcare provider for proper interpretation.")
                    
                    # Clear test input from session state after prediction
                    if 'test_input' in st.session_state:
                        del st.session_state['test_input']
                        
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    st.exception(e)
                    
                    # Try to diagnose the issue
                    st.write("**Debug information:**")
                    st.write(f"Feature names from model: {feature_names}")
                    st.write(f"Input data columns: {input_data.columns.tolist()}")
                    st.write(f"Categorical features: {list(feature_encoders.keys())}")
                    st.write(f"Numerical features: {numerical_cols}")
        else:
            st.error("Failed to load model. Please check if the Excel file exists and has the correct format.")

if __name__ == "__main__":
    main()

