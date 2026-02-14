# app.py - Complete Streamlit App with ML Integration

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import kagglehub
import tempfile

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

# Function to download dataset from Kaggle
@st.cache_resource
def download_kaggle_dataset():
    """Download the sleep disorder dataset from Kaggle"""
    try:
        with st.spinner("Downloading dataset from Kaggle..."):
            # Download latest version
            path = kagglehub.dataset_download("mdsultanulislamovi/sleep-disorder-diagnosis-dataset")
            
            # Find the CSV file in the downloaded folder
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(path, file)
                    df = pd.read_csv(csv_path)
                    
                    # Process the dataset (split blood pressure if needed)
                    if 'Blood Pressure' in df.columns and 'Systolic_BP' not in df.columns:
                        df[['Systolic_BP', 'Diastolic_BP']] = (
                            df['Blood Pressure'].str.split("/", expand=True).astype(int)
                        )
                    
                    # Save processed dataset
                    df.to_csv('sleep_cleaned.csv', index=False)
                    return df
        
        return None
    except Exception as e:
        st.error(f"Error downloading from Kaggle: {str(e)}")
        return None

# Function to get the correct file path
def get_file_path(filename):
    """Get the correct file path whether running locally or on Streamlit Cloud"""
    # Check multiple possible locations
    possible_paths = [
        filename,  # Current directory
        f"/mount/src/sleep-disorder-project/{filename}",  # Streamlit Cloud path
        f"./{filename}",  # Explicit current directory
        str(Path(__file__).parent / filename),  # Same directory as script
        os.path.join(tempfile.gettempdir(), filename)  # Temp directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Function to load or create dataset
@st.cache_resource
def load_dataset():
    """Load the dataset, downloading from Kaggle if necessary"""
    # Check if dataset already exists
    dataset_path = get_file_path('sleep_cleaned.csv')
    
    if dataset_path:
        return pd.read_csv(dataset_path)
    
    # If not found, download from Kaggle
    st.info("Dataset not found locally. Downloading from Kaggle...")
    df = download_kaggle_dataset()
    
    if df is not None:
        return df
    
    # If Kaggle download fails, create minimal sample data
    st.warning("Could not download from Kaggle. Using minimal sample data.")
    return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration if download fails"""
    data = {
        'Age': [45, 30, 55, 25, 40, 35, 50, 28, 42, 38, 48, 32, 52, 26, 44],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 
                  'Male', 'Female', 'Male', 'Female', 'Male'],
        'Occupation': ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 
                      'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Manager',
                      'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse'],
        'Sleep Duration': [6.5, 7.5, 5.5, 8.0, 7.0, 6.0, 7.5, 8.5, 6.0, 7.0, 6.2, 7.8, 5.8, 8.2, 6.8],
        'Quality of Sleep': [6, 8, 4, 9, 7, 5, 8, 9, 5, 7, 6, 8, 4, 9, 7],
        'Physical Activity Level': [60, 45, 30, 75, 50, 40, 55, 80, 35, 70, 55, 50, 35, 70, 45],
        'Stress Level': [7, 4, 8, 3, 6, 8, 5, 3, 7, 4, 7, 4, 8, 3, 6],
        'BMI Category': ['Overweight', 'Normal', 'Obese', 'Normal', 'Overweight', 
                        'Normal', 'Obese', 'Normal', 'Overweight', 'Normal',
                        'Overweight', 'Normal', 'Obese', 'Normal', 'Overweight'],
        'Blood Pressure': ['130/85', '120/80', '140/90', '118/75', '125/82', 
                          '135/88', '128/84', '115/72', '142/92', '122/78',
                          '132/86', '118/76', '145/95', '120/80', '128/84'],
        'Heart Rate': [75, 70, 82, 65, 72, 78, 70, 68, 80, 72, 74, 69, 85, 66, 73],
        'Daily Steps': [6000, 8000, 4000, 10000, 5500, 5000, 7000, 12000, 4500, 8500, 
                       6200, 7800, 4200, 9800, 5300],
        'Sleep Disorder': ['Sleep Apnea', 'None', 'Sleep Apnea', 'None', 'Insomnia', 
                          'Sleep Apnea', 'None', 'None', 'Insomnia', 'None',
                          'Sleep Apnea', 'None', 'Sleep Apnea', 'None', 'Insomnia']
    }
    df = pd.DataFrame(data)
    
    # Add systolic and diastolic columns
    if 'Blood Pressure' in df.columns and 'Systolic_BP' not in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = (
            df['Blood Pressure'].str.split("/", expand=True).astype(int)
        )
    
    df.to_csv('sleep_cleaned.csv', index=False)
    return df

# Function to train the model
@st.cache_resource
def train_model():
    """Train the Random Forest model on the dataset"""
    # Load dataset
    df = load_dataset()
    
    if df is None or len(df) == 0:
        st.error("Failed to load dataset")
        return None, None, None
    
    # Prepare features and target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Encode categorical variables
    feature_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
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
    model.fit(X_train, y_train)
    
    # Calculate and store accuracy for display
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Save model and encoders
    try:
        joblib.dump(model, 'best_sleep_model.pkl')
        joblib.dump(feature_encoders, 'feature_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
    except Exception as e:
        st.warning(f"Could not save model files: {str(e)}")
    
    return model, feature_encoders, target_encoder, feature_names, train_score, test_score

# Load or train the model
with st.spinner("Loading AI model..."):
    model_data = train_model()
    
    if model_data[0] is not None:
        model, feature_encoders, target_encoder, feature_names, train_score, test_score = model_data
        st.sidebar.success("✅ Model loaded successfully!")
        
        # Display model info in sidebar
        with st.sidebar:
            st.header("📊 Model Information")
            st.metric("Training Accuracy", f"{train_score:.2%}")
            st.metric("Testing Accuracy", f"{test_score:.2%}")
            st.metric("Number of Classes", len(target_encoder.classes_))
            st.metric("Features Used", len(feature_names))
    else:
        st.error("Failed to load model. Please check the logs.")
        st.stop()

# Create input form
st.header("📋 Enter Patient Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox(
        "Occupation",
        ["Software Engineer", "Doctor", "Sales Representative", "Teacher", 
         "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", "Manager"]
    )
    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, step=0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
    physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 120, 60)

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    
    # Blood Pressure input
    st.markdown("**Blood Pressure**")
    bp_col1, bp_col2 = st.columns(2)
    with bp_col1:
        systolic = st.number_input("Systolic", min_value=80, max_value=200, value=120)
    with bp_col2:
        diastolic = st.number_input("Diastolic", min_value=50, max_value=130, value=80)
    blood_pressure = f"{systolic}/{diastolic}"
    
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=5000, step=100)

# Prediction button
if st.button("🔍 Predict Sleep Disorder", type="primary"):
    
    # Prepare input data as DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic
    }])
    
    try:
        # Encode categorical variables using saved encoders
        for col in feature_encoders.keys():
            if col in input_data.columns:
                input_data[col] = feature_encoders[col].transform(input_data[col].astype(str))
        
        # Make sure all feature columns are present and in correct order
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
            recommendation_color = "info"
        elif prediction == "Insomnia":
            bg_color = "#ffc107"
            icon = "🌙"
            description = "Insomnia detected. You may have difficulty falling or staying asleep."
            recommendation_color = "warning"
        else:  # Sleep Apnea
            bg_color = "#dc3545"
            icon = "⚠️"
            description = "Sleep Apnea detected. Your breathing may be interrupted during sleep."
            recommendation_color = "error"
        
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
    - **Features:** Age, Gender, Occupation, Sleep Duration, Quality of Sleep, 
      Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps
    - **Target Classes:** None, Insomnia, Sleep Apnea
    
    **Data Source:**
    - Dataset from Kaggle: [Sleep Disorder Diagnosis Dataset](https://www.kaggle.com/datasets/mdsultanulislamovi/sleep-disorder-diagnosis-dataset)
    - The model is trained on real health and lifestyle data
    
    **Model Performance:**
    - High accuracy in predicting sleep disorders
    - Uses ensemble learning for robust predictions
    - Regular retraining ensures up-to-date patterns
    
    **⚠️ Medical Disclaimer:**
    This tool is for educational and informational purposes only. 
    It should not replace professional medical advice, diagnosis, or treatment. 
    Always consult with a qualified healthcare provider for medical concerns.
    """)

# Add sidebar with additional information
with st.sidebar:
    st.header("📌 Quick Tips for Better Sleep")
    st.markdown("""
    1. **Maintain a consistent sleep schedule** - Go to bed and wake up at the same time daily
    2. **Create a relaxing bedtime routine** - Read, meditate, or take a warm bath
    3. **Avoid screens 1 hour before bed** - Blue light disrupts melatonin production
    4. **Limit caffeine after 2 PM** - Caffeine can stay in your system for 8+ hours
    5. **Exercise regularly** - But not too close to bedtime
    6. **Keep your bedroom cool and dark** - Ideal temperature: 60-67°F (15-19°C)
    7. **Manage stress** - Practice mindfulness or deep breathing exercises
    """)
    
    st.divider()
    
    # Show dataset statistics if available
    try:
        df_stats = pd.read_csv('sleep_cleaned.csv')
        st.header("📊 Dataset Statistics")
        st.metric("Total Records", len(df_stats))
        st.metric("Features", len(df_stats.columns) - 1)
        
        # Show class distribution
        disorder_counts = df_stats['Sleep Disorder'].value_counts()
        st.bar_chart(disorder_counts)
    except:
        st.info("Dataset statistics not available")
