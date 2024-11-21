import os 
import glob
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import math
import yaml
from src.prediction.prediction_utils import predict_single_image


# Define global CSS styles
st.markdown("""
<style>
.title {
        font-size: 40px;
        background: -webkit-linear-gradient(45deg, #007BFF, #4285F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-family: 'Helvetica', sans-serif;
        margin-bottom: 30px;
        border-bottom: 4px solid #162a3a; /* Secondary background color */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); /* Soft text shadow */
    }
    .subheader {
        font-size: 26px;
        color: #95D2B3;
        margin-top: 25px;
        font-weight: bold;
        font-family: 'Helvetica', sans-serif;
        border-bottom: 2px solid #162a3a; /* Secondary background color */
        padding-bottom: 5px;
    }
    h4 {
    font-size: 20px;
    color: #F4A261; /* Dark text color for h4 */
    font-weight: bold;
    font-family: 'Helvetica', sans-serif;
    border-bottom: 2px solid #C80036;
    margin-top: 15px;
    margin-bottom: 10px;
    }              
    .body-text {
        font-size: 18px;
        color: #333333; /* Dark text color for readability */
        font-family: 'Helvetica', sans-serif;
        line-height: 1.6; /* Improved line spacing */
    }
.prediction-green {
    color: #41B06E;
    font-size: 24px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}
.prediction-red {
    color: #FF204E;
    font-size: 24px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}
.confidence-green {
    color: #41B06E;
    font-size: 24px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}
.confidence-yellow {
    color: yellow;
    font-size: 24px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}
.confidence-red {
    color: #FF204E;
    font-size: 24px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)


script_directory = os.path.dirname(os.path.abspath(__file__))

# Function to get list of image files from a directory (case insensitive)
def get_image_options(path):
    total_path = os.path.join(script_directory, path)

    image_extensions = ['jpg', 'jpeg', 'png']
    image_files = [f for f in os.listdir(total_path) if f.split('.')[-1].lower() in image_extensions]
    return image_files

def display_image(image):

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if image is not None:
            st.image(image, caption="Selected Image", use_column_width=True)

def display_prediction(prediction, confidence):
    col1, col2, col3, col4 = st.columns([2, 3,  2, 2])
    with col2:
        st.write("###### **Prediction:**")
        if prediction == "TB Detected":
            st.markdown(f"<p class='prediction-red'>{prediction}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='prediction-green'>{prediction}</p>", unsafe_allow_html=True)

    with col3:
        st.write("###### **Confidence:**")
        if confidence > 85:
            st.markdown(f"<p class='confidence-green'>{confidence}%</p>", unsafe_allow_html=True)
        elif 65 <= confidence <= 85:
            st.markdown(f"<p class='confidence-yellow'>{confidence}%</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='confidence-red'>{confidence}%</p>", unsafe_allow_html=True)

# Function to display prediction insights on Streamlit
def display_post_prediction_insights(prediction):
    path = 'src/prediction/post-prediction-info.yaml'
    post_pred_info_path = os.path.join(script_directory, path)
    try:
        with open(post_pred_info_path,'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")  


    if prediction == 'TB Detected':
        disease_info = data['tb']
        
        st.markdown('<h4>Actionable Insights</h4>', unsafe_allow_html=True)
        # Display symptoms
        st.write("**SYMPTOMS:**")
        for symptom in disease_info.get("symptoms",[]):
            st.write(f"- {symptom}")

        st.write("")

        # Display treatment
        st.write("\n**TREATMENT MEASURES:**")
        for treatment in disease_info.get("treatment_measures", []):
            st.write(f"- {treatment}")
        
    elif prediction == 'TB Not Detected':
        disease_info = data['non_tb']
        
        st.markdown('<h4>Actionable Insights</h4>', unsafe_allow_html=True)
        # Display further tests
        st.write("**FURTHER TESTS REQUIRED:**")
        for test in disease_info.get("further_tests",[]):
            st.write(f"- {test}")
        
        st.write("")    

        # Display precautions
        st.write("\n**PRECAUTIONS:**")
        for precaution in disease_info.get("precautions", []):
            st.write(f"- {precaution}")
    else:
        st.warning("No information available.")


def main():
    IMAGE_HEIGHT = 232
    IMAGE_WIDTH = 232

    st.markdown('<p class="title">Diagnostics Application</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload Image</p>', unsafe_allow_html=True)
    st.text("ℹ️ Focus on the lungs, keep the area of interest at the center of the image for better results")

    # Create a container for the "Upload Image" section
    with st.container(border=True):
        col1, col2 = st.columns(2)

        # Column 1: File uploader
        with col1:
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        # Column 2: Select image from selectbox
        with col2:
            image_options = get_image_options('examples')
            if len(image_options) > 0:
                selected_image = st.selectbox("Select an Image", image_options, index=None)

    # Get the image
    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    elif selected_image is not None:
        
        image_path = os.path.join(script_directory,'examples', selected_image)
        image = load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    else:
        st.write("No image available.")
        return

    with st.container(border=True):
        display_image(image)
    
    # Define prediction variables globally
    global prediction, prediction_probability
    prediction = None
    prediction_probability = None

    if st.button("Predict"):
        file_path = glob.glob(os.path.join(script_directory,"src/models", "*.h5"))[0]

        # Load the model
        model = load_model(file_path)

        #image to array
        image_array = img_to_array(image)


        # Predict disease and prediction probability
        prediction, prediction_probability = predict_single_image(image_arr=image_array, model=model)
        if prediction == 'Unknown':
            st.error(" ⚠️ Unable to identify, Please upload another image ")
            prediction = None

        with st.container(border=True):
            if prediction is not None and prediction_probability is not None:
                confidence = prediction_probability * 100
                trimmed_confidence = math.floor(confidence * 100) / 100
                display_prediction(prediction, trimmed_confidence)
            else:
                st.write("No prediction available.")
                
        # Update the prediction variable globally
        prediction = prediction
        prediction_probability = prediction_probability

        st.warning('This is an AI prediction - Further tests and symptoms checks are needed to confirm the diagnosis', icon="⚠️")

    if prediction is not None:
        with st.container(border=True):
            display_post_prediction_insights(prediction)
          
# if __name__ == "__main__":
main()

