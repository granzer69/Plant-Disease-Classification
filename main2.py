import streamlit as st
import tensorflow as tf
import numpy as np

st.markdown("""
<style>
.main-header {
    color: #2C5F2D;
    text-align: center;
    font-weight: bold;
}
.disease-result {
    background-color: #F0F4F0;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
.recommendation {
    color: #4A6741;
}
</style>
""", unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return predicted_class_index, confidence

# Disease Recommendations Dictionary
disease_recommendations = {
    'Apple___Apple_scab': 'Apply fungicides (e.g., captan, myclobutanil). Prune infected branches and improve airflow.',
    'Apple___Black_rot': 'Remove infected fruits and cankers. Apply fungicides (e.g., thiophanate-methyl).',
    'Apple___Cedar_apple_rust': 'Remove nearby juniper trees. Apply fungicides (e.g., myclobutanil).',
    'Apple___healthy': 'No action needed. Maintain general care.',
    'Blueberry___healthy': 'No action needed. Maintain proper watering and nutrient management.',
    'Cherry_(including_sour)___Powdery_mildew': 'Use fungicides (e.g., sulfur, potassium bicarbonate). Increase airflow through pruning.',
    'Cherry_(including_sour)___healthy': 'No action needed.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides (e.g., strobilurins, triazoles). Rotate crops and use resistant varieties.',
    'Corn_(maize)___Common_rust_': 'Plant resistant hybrids. Use fungicides (e.g., propiconazole).',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids and apply fungicides if needed.',
    'Corn_(maize)___healthy': 'No action needed.',
    'Grape___Black_rot': 'Remove infected plant parts. Apply fungicides (e.g., myclobutanil).',
    'Grape___Esca_(Black_Measles)': 'Prune and destroy infected wood. Avoid mechanical injuries.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply copper-based fungicides. Ensure proper canopy management.',
    'Grape___healthy': 'No action needed.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Remove infected trees. Control Asian citrus psyllid using insecticides.',
    'Peach___Bacterial_spot': 'Use copper-based sprays. Remove and destroy infected leaves and fruits.',
    'Peach___healthy': 'No action needed.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper fungicides. Avoid overhead watering.',
    'Pepper,_bell___healthy': 'No action needed.',
    'Potato___Early_blight': 'Use fungicides (e.g., chlorothalonil, mancozeb). Remove plant debris.',
    'Potato___Late_blight': 'Apply fungicides (e.g., metalaxyl). Destroy infected plants.',
    'Potato___healthy': 'No action needed.',
    'Raspberry___healthy': 'No action needed.',
    'Soybean___healthy': 'No action needed.',
    'Squash___Powdery_mildew': 'Apply fungicides (e.g., sulfur, potassium bicarbonate). Ensure good airflow around plants.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Apply fungicides (e.g., captan). Avoid overhead irrigation.',
    'Strawberry___healthy': 'No action needed.',
    'Tomato___Bacterial_spot': 'Use copper-based sprays. Remove infected plants.',
    'Tomato___Early_blight': 'Apply fungicides (e.g., chlorothalonil). Rotate crops and remove debris.',
    'Tomato___Late_blight': 'Use fungicides (e.g., metalaxyl). Destroy infected plants.',
    'Tomato___Leaf_Mold': 'Increase airflow and avoid overhead watering. Use fungicides if needed.',
    'Tomato___Septoria_leaf_spot': 'Apply fungicides (e.g., mancozeb, chlorothalonil). Remove infected leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray water to wash off mites. Use insecticidal soap or neem oil.',
    'Tomato___Target_Spot': 'Remove infected plant parts. Apply fungicides if needed.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies (vector). Use resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected plants. Disinfect tools and avoid handling plants excessively.',
    'Tomato___healthy': 'No action needed.'
}

def get_disease_recommendation(disease):
    return disease_recommendations.get(disease, 'Consult a local agricultural expert for specific treatment.')

# Sidebar and App Modes
st.sidebar.title("🌿 Plant Health Dashboard")
app_mode = st.sidebar.selectbox("Navigate", ["Home", "Disease Recognition", "About"])

if app_mode == "Home":
    st.markdown("<h1 class='main-header'>Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    ### Quick Plant Health Insights
    - Upload plant leaf images
    - Get instant disease detection
    - Receive expert recommendations
    """)

elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='main-header'>Detect Plant Diseases</h1>", unsafe_allow_html=True)
    
    # Image Upload
    test_image = st.file_uploader("Upload Plant Leaf Image", type=['jpg', 'png', 'jpeg'])
    
    if test_image is not None:
        # Display uploaded image
        st.image(test_image, width=300, caption='Uploaded Plant Leaf')
        
        # Prediction Button
        if st.button("Analyze Leaf"):
            st.snow()
            with st.spinner('Analyzing Image...'):
                result_index, confidence = model_prediction(test_image)
                
                # Class Names
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy'
                ]
                
                detected_disease = class_name[result_index]  # Ensure this variable is defined here
                recommendation = get_disease_recommendation(detected_disease)
                
                # Display Results
                st.markdown(
                    f"""
                    <div class='disease-result'>
                        <p><strong style="color: black;">Disease:</strong> <span style="color: black;">{detected_disease}</p>
                        <p><strong style="color: black;">Confidence:</strong><span style="color: black;"> {confidence:.2f}%</p>
                        <p class='recommendation'><strong style="color: black">Recommendation:</strong> <span style="color: black;">{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


elif app_mode == "About":
    st.markdown("<h1 class='main-header'>About Our System</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### Our Mission
    Empowering farmers and gardeners with quick, accurate plant disease detection.

    ### Dataset Details
    - 87K RGB images of crop leaves
    - 38 different plant disease classes
    - 80/20 training and validation split
    """)

    st.markdown("""
<div style='text-align: center; margin-top: 100px; color: white;'>
    <p>Made by Ribhu S, Saketh and Varun</p>
</div>
""", unsafe_allow_html=True)
