# Tuberculosis Detection Using Lungs X-Ray Images and Multivariate Data

This project is a web application that leverages machine learning techniques to detect tuberculosis (TB) from X-ray images. The application provides a user-friendly interface for uploading X-ray images and receiving instant diagnostic predictions.


## Features

- Upload X-ray images in various formats.
- Real-time prediction of tuberculosis presence.
- User-friendly interface built with Streamlit.
- Visual feedback on the prediction results.

## Future Programs and Directions
- 

## Usage

You can use the application directly at the following link:


### Steps to Use the Application:

1. Open the link in your web browser.
2. Upload an X-ray image using the provided file uploader or select from the pre-existing images from the drop down menu.
3. Click on the "Predict" button to receive the results.
4. Review the output, which indicates the presence or absence of tuberculosis.


## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/iamamofa/TB-detection.git
   cd TB-detection/tb-detection-app
2. **Create a virtual environment (optional but recommended):**

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install the required packages:**

   ```bash
    pip install -r requirements.txt
4. **Run the application:**

   ```bash
   streamlit run app.py
5. Open your web browser and navigate to http://localhost:8501 and follow 'Steps to Use the Application' instructions above.


## Technologies Used
-   Python
-   Streamlit
-   TensorFlow - Keras
-   OpenCV (for image processing)
-    NumPy / Pandas (for data handling)
