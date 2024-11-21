import streamlit as st


st.set_page_config(page_title="Chest Xray Tuberculosis Prediction", 
                   page_icon=":material/thumb_up:", 
                   layout="wide",
                   initial_sidebar_state="expanded",)



about_page = st.Page("about.py", title="About", icon=":material/info:")
app_page = st.Page("prediction_application.py", title="Application", icon=":material/image_search:")

pg = st.navigation([about_page, app_page])

pg.run()