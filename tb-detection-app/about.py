import streamlit as st

# Styling the header and subheaders
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        background: -webkit-linear-gradient(45deg,#007BFF, #4285F4);
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
        border-bottom: 2px solid #162a3a;; /* Secondary background color */
        padding-bottom: 5px;
    }
    .body-text {
        font-size: 20px;
        color: #333333; /* Dark text color for readability */
        font-family: 'Helvetica', sans-serif;
        line-height: 1.6; /* Improved line spacing */
    }
    </style>
    """, unsafe_allow_html=True)


# Title
st.markdown('<p class="title">Chest MycoBacterium Tuberculosis Detection</p>', unsafe_allow_html=True)

# Additional information
st.markdown("""[For more information about Tuberculosis in Ghana, visit - [PANGenS](https://pangens.org/tb-genomic-surveillance/)]""")


# Introduction Subheader
st.markdown('<p class="subheader">Introduction</p>', unsafe_allow_html=True)

# Introduction Body
st.markdown("""
Welcome to the Chest X-ray Tuberculosis Detection platform! This application leverages advanced 
Convolutional Neural Networks (CNNs) to assist healthcare professionals in diagnosing diseases
from chest X-rays, particularly focusing on detecting Tuberculosis.
Our goal is to improve diagnostic accuracy and provide  quick insights to aid clinical decision-making.
""")

# Project Description and Our Work Subheader
st.markdown('<p class="subheader">Project Description </p>', unsafe_allow_html=True)

# Project Description and Our Work Body
st.markdown("""
Chest X-rays are a widely used diagnostic tool, but interpreting them requires significant expertise.
Our system is designed to automatically analyze chest X-ray images and detect signs of Tuberculosis
to assist radiologists and medical professionals. By using cutting-edge deep learning models like EfficientNet 
and DenseNet, we aim to deliver accurate predictions and support in early detection of Tuberculosis
""")

# List of Crops
#st.markdown("""
#- **Penumonia**
#- **NA**
#""")

st.markdown('<p class="subheader">Future Work</p>', unsafe_allow_html=True)


st.markdown("""
We are committed to continuously enhancing our application to better serve the Healthcare community. Our future work includes:

- **Expanding Chest Xray Database**: Adding support for more diseases.\n
- **Develop Model Training Tool** : Developing an intuitive UI to facilitate deep learning model training with labeled Chest Xray disease image data.\n           
- **Cost-Sensitive Learning**: Prioritizing accurate detection of additional critical diseases by assigning higher\n
  penalties to misclassifications with greater consequences. This can improve overall diagnostic accuracy and aid decision-making.\n
- **Mobile Application**: Creating a mobile app version for on-the-go access and usability.\n
- **Multilingual Support**: Providing support in various languages to cater to a global audience."
""")
