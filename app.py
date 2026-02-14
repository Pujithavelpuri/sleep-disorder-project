# app.py - Complete Streamlit App with ML Integration using Excel file

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
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
    
    **Option 3: Install directly in the app**
    ```python
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
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
            st.sidebar.success(f"✅ Loaded dataset from {excel_file}")
            
            # Display basic info about the dataset
            st.sidebar.write(f"**Rows:** {len(df)}")
            st.sidebar.write(f"**Columns:** {len(df.columns)}")
            st.sidebar.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Process the dataset (split blood pressure if needed)
            if 'Blood Pressure' in df.columns and 'Systolic_BP' not in df.columns:
                try:
                    df[['Systolic_BP', 'Diastolic_BP']] = (
                        df['Blood Pressure'].str.split("/", expand=True).astype(int)
                    )
                    st.sidebar.info("✓ Split Blood Pressure into Systolic and Diastolic")
                except Exception as e:
                    st.sidebar.warning(f"Could not split Blood Pressure: {str(e)}")
            
            # Save as CSV for backup and faster loading next time
            df.to_csv('sleep_cleaned.csv', index=False)
            st.sidebar.info("✓ Saved backup as CSV")
            
            return df
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            st.info("Trying alternative method...")
            
            # Try alternative method without specifying engine
            try:
                df = pd.read_excel(excel_file)
                return df
            except:
                return None
    else:
        st.error(f"❌ Excel file '{excel_file}' not found in root directory!")
        st.info("Please ensure 'sleep_disorder_powerbi.xlsx' is in the same directory as this app.")
        
        # Show current directory contents
        st.write("Files in current directory:")
        files = os.listdir('.')
        for file in files:
            st.write(f"- {file}")
        return None

# Alternative function to load CSV if Excel fails
@st.cache_resource
def load_csv_dataset():
    """Load the sleep disorder dataset from CSV file (backup)"""
    csv_file = "sleep_cleaned.csv"
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            st.sidebar.success(f"✅ Loaded dataset from {csv_file} (backup)")
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    return None

# Function to train the model
@st.cache_resource
def train_model():
    """Train the Random Forest model on the dataset"""
    
    # First try to load from Excel
    df = load_excel_dataset()
    
    # If Excel fails, try CSV backup
    if df is None:
        st.warning("Trying to load from CSV backup...")
        df = load_csv_dataset()
    
    # If both fail, show error
    if df is None:
        st.error("Failed to load dataset from any source")
        return None, None, None, None, None, None
    
    # Check if 'Sleep Disorder' column exists
    if 'Sleep Disorder' not in df.columns:
        st.error("Column 'Sleep Disorder' not found in the dataset!")
        st.write("Available columns:", df.columns.tolist())
        
        # Look for similar column names
        possible_targets = [col for col in df.columns if 'sleep' in col.lower() or 'disorder' in col.lower()]
        if possible_targets:
            st.info(f"Did you mean one of these columns? {possible_targets}")
        return None, None, None, None, None, None
    
    # Prepare features and target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Display dataset info in main area
    st.success("✅ Dataset loaded successfully!")
    
    with st.expander("📊 Dataset Overview"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len(feature_names))
        col3.metric("Target Classes", y.nunique())
        
        st.write("**First few rows:**")
        st.dataframe(df.head())
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode categorical variables
    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Display class distribution
    st.write("**Class Distribution:**")
    class_dist = pd.DataFrame({
        'Class': target_encoder.classes_,
        'Count': pd.Series(y_encoded).value_counts().sort_index().values
    })
    st.dataframe(class_dist)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train the model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
    
    # Calculate and store accuracy for display
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model and encoders
    try:
        joblib.dump(model, 'best_sleep_model.pkl')
        joblib.dump(feature_encoders, 'feature_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        st.sidebar.success("✅ Model saved successfully!")
    except Exception as e:
        st.sidebar.warning(f"Could not save model files: {str(e)}")
    
    return model, feature_encoders, target_encoder, feature_names, train_score, test_score, feature_importance

# Main app
def main():
    # Load or train the model
    with st.spinner("Loading AI model and training on Excel data..."):
        model_data = train_model()
        
        if model_data[0] is not None:
            model, feature_encoders, target_encoder, feature_names, train_score, test_score, feature_importance = model_data
            
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
            numerical_features = [col for col in feature_names if col not in categorical_features and col not in ['Systolic_BP', 'Diastolic_BP']]
            bp_features = ['Systolic_BP', 'Diastolic_BP'] if all(f in feature_names for f in ['Systolic_BP', 'Diastolic_BP']) else []
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            input_dict = {}
            
            with col1:
                # Age input (if present)
                if 'Age' in numerical_features:
                    input_dict['Age'] = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
                
                # Gender input (if present)
                if 'Gender' in categorical_features:
                    gender_options = feature_encoders['Gender'].classes_.tolist() if 'Gender' in feature_encoders else ["Male", "Female"]
                    input_dict['Gender'] = st.selectbox("Gender", gender_options)
                
                # Occupation input (if present)
                if 'Occupation' in categorical_features:
                    occupation_options = feature_encoders['Occupation'].classes_.tolist() if 'Occupation' in feature_encoders else []
                    input_dict['Occupation'] = st.selectbox("Occupation", occupation_options)
                
                # Sleep Duration (if present)
                if 'Sleep Duration' in numerical_features:
                    input_dict['Sleep Duration'] = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, step=0.1)
                
                # Quality of Sleep (if present)
                if 'Quality of Sleep' in numerical_features:
                    input_dict['Quality of Sleep'] = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
                
                # Physical Activity Level (if present)
                if 'Physical Activity Level' in numerical_features:
                    input_dict['Physical Activity Level'] = st.slider("Physical Activity Level (minutes/day)", 0, 120, 60)
            
            with col2:
                # Stress Level (if present)
                if 'Stress Level' in numerical_features:
                    input_dict['Stress Level'] = st.slider("Stress Level (1-10)", 1, 10, 5)
                
                # BMI Category (if present)
                if 'BMI Category' in categorical_features:
                    bmi_options = feature_encoders['BMI Category'].classes_.tolist() if 'BMI Category' in feature_encoders else ["Normal", "Overweight", "Obese"]
                    input_dict['BMI Category'] = st.selectbox("BMI Category", bmi_options)
                
                # Blood Pressure input (if systolic and diastolic are present)
                if bp_features:
                    st.markdown("**Blood Pressure**")
                    bp_col1, bp_col2 = st.columns(2)
                    with bp_col1:
                        systolic = st.number_input("Systolic", min_value=80, max_value=200, value=120)
                    with bp_col2:
                        diastolic = st.number_input("Diastolic", min_value=50, max_value=130, value=80)
                    input_dict['Blood Pressure'] = f"{systolic}/{diastolic}"
                    input_dict['Systolic_BP'] = systolic
                    input_dict['Diastolic_BP'] = diastolic
                
                # Heart Rate (if present)
                if 'Heart Rate' in numerical_features:
                    input_dict['Heart Rate'] = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
                
                # Daily Steps (if present)
                if 'Daily Steps' in numerical_features:
                    input_dict['Daily Steps'] = st.number_input("Daily Steps", min_value=0, max_value=20000, value=5000, step=100)
            
            # Prediction button
            if st.button("🔍 Predict Sleep Disorder", type="primary"):
                
                # Create DataFrame
                input_data = pd.DataFrame([input_dict])
                
                try:
                    # Encode categorical variables using saved encoders
                    for col in feature_encoders.keys():
                        if col in input_data.columns:
                            # Handle unknown categories
                            try:
                                input_data[col] = feature_encoders[col].transform(input_data[col].astype(str))
                            except ValueError:
                                # If category not seen during training, use the most common class
                                st.warning(f"Unknown {col} value. Using default encoding.")
                                input_data[col] = 0  # Default to first class
                    
                    # Make sure all feature columns are present and in correct order
                    for col in feature_names:
                        if col not in input_data.columns:
                            input_data[col] = 0  # Add missing columns with default value
                    
                    input_data = input_data[feature_names]
                    
                    # Make prediction
                    prediction_encoded = model.predict(input_data)[0]
                    prediction_proba = model.predict_proba(input_data)[0]
                    
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
                    else:  # Sleep Apnea
                        bg_color = "#dc3545"
                        icon = "⚠️"
                        description = "Sleep Apnea detected. Your breathing may be interrupted during sleep."
                    
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
                    else:  # Sleep Apnea
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
                        
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    st.info("Please ensure all fields are filled correctly.")
            
            # Add information about the model and data source
            with st.expander("ℹ️ About the Model & Data"):
                st.markdown("""
                **Model Information:**
                - **Algorithm:** Random Forest Classifier (Ensemble Learning)
                - **Data Source:** sleep_disorder_powerbi.xlsx (Excel file from root directory)
                - **Features Used:** Various health and lifestyle indicators
                - **Target Classes:** None, Insomnia, Sleep Apnea
                
                **Model Performance:**
                - Trained on real health and lifestyle data
                - Uses ensemble learning for robust predictions
                - Feature importance analysis shows key predictors
                
                **⚠️ Medical Disclaimer:**
                This tool is for educational and informational purposes only. 
                It should not replace professional medical advice, diagnosis, or treatment. 
                Always consult with a qualified healthcare provider for medical concerns.
                """)
        else:
            st.error("Failed to load model. Please check if the Excel file exists and has the correct format.")

if __name__ == "__main__":
    main()
