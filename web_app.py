import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scripts.model_utils import load_model
from scripts.prediction_utils import get_prediction_details

# Configuration
CONFIDENCE_THRESHOLD = 90.0
REJECTION_THRESHOLD = 10.0

# Page config
st.set_page_config(
    page_title="Fruit & Vegetable Quality Checker",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
        font-size: 24px;
    }
    .fresh {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .rotten {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeeba;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

def sidebar_info():
    st.sidebar.image("assets/profile_pic.jpg", use_container_width=True)
    st.sidebar.title("About the Author")
    st.sidebar.markdown("**Divyansh Singh**")
    st.sidebar.markdown(
        """
        Connect with me:
        - [LinkedIn](https://www.linkedin.com/in/divyansh-singh-26a95a248/)
        - [GitHub](https://github.com/di84867)
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This application uses Deep Learning to detect the quality of fruits and vegetables.")

def main():
    sidebar_info()
    st.title("🍎 Fruit & Vegetable Quality Checker")
    st.markdown("---")

    option = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))
    
    model = get_model()
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button("Analyze Quality"):
                with st.spinner('Analyzing...'):
                    # Preprocess
                    img_array = np.array(image.convert('RGB'))
                    img_array = cv2.resize(img_array, (224, 224))
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    # Predict
                    is_valid, details = get_prediction_details(
                        model, img_array, CONFIDENCE_THRESHOLD, REJECTION_THRESHOLD
                    )
                    
                    # Display Results
                    if is_valid:
                        result_class = "fresh" if "Fresh" in details['label'] else "rotten"
                        st.markdown(f"""
                            <div class="result-box {result_class}">
                                Prediction: {details['label']}<br>
                                Confidence: {details['confidence']:.2f}%
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="result-box warning">
                                ⚠️ Classification Uncertain<br>
                                Please ensure the image contains a single fruit or vegetable.
                            </div>
                        """, unsafe_allow_html=True)
                        st.info("Try a clearer image or different angle.")

    elif option == "Use Camera":
        img_file_buffer = st.camera_input("Take a picture")
        
        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Preprocess
            img_array = cv2.resize(cv2_img, (224, 224))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Correct color for model
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            is_valid, details = get_prediction_details(
                model, img_array, CONFIDENCE_THRESHOLD, REJECTION_THRESHOLD
            )
            
            if is_valid:
                result_class = "fresh" if "Fresh" in details['label'] else "rotten"
                st.markdown(f"""
                    <div class="result-box {result_class}">
                        Prediction: {details['label']}<br>
                        Confidence: {details['confidence']:.2f}%
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-box warning">
                        ⚠️ Classification Uncertain<br>
                        Please ensure the image contains a single fruit or vegetable.
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
