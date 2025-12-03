import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import DenseNet201

# ===============
# 1. PAGE CONFIG
# ===============
st.set_page_config(
    page_title="Image Captioning",
    page_icon=None,
    layout="wide"
)


st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0F1116; /* Deep matte black/blue */
        color: #E6E6E6;
    }

    /* Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Header Styling */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    
    /* Clean Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #2D3748;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: transparent;
        border: none;
        color: #718096;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #A0AEC0;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #63B3ED; /* Soft Blue */
        border-bottom: 2px solid #63B3ED;
    }

    /* Primary Button Styling */
    .stButton>button {
        background-color: #3182CE; /* Professional Blue */
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2B6CB0;
    }

    /* Card/Container Styling */
    .css-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    .caption-text {
        font-size: 1.5rem;
        font-weight: 300;
        color: #E2E8F0;
        border-left: 3px solid #63B3ED;
        padding-left: 15px;
        margin-top: 20px;
    }
    
    /* Remove default image margin */
    .stImage {
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ===============
# 2. MODEL
# ===============

MODEL_PATH = 'models/model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'
FLICKR_FOLDER = 'assets/flickr8k_images/'
MAX_LENGTH = 37 

@st.cache_resource
def load_resources():
    try:
        caption_model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        base_model = DenseNet201(weights="imagenet")
        feature_extractor = Model(base_model.input, base_model.layers[-2].output)
        return caption_model, tokenizer, feature_extractor
    except Exception as e:
        return None, None, None

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img = img_to_array(img)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

def predict_caption(model, tokenizer, feature_extractor, image):
    processed_img = preprocess_image(image)
    feature = feature_extractor.predict(processed_img, verbose=0)
    
    in_text = "startseq"
    for i in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        yhat = model.predict({'image_input': feature, 'text_input': sequence}, verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None: break
        in_text += " " + word
        if word == "endseq": break
            
    final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return final_caption.capitalize()

# ===============
# 3. UI LAYOUT
# ===============

caption_model, tokenizer, feature_extractor = load_resources()


st.title("Image Captioning")
st.markdown("<p style='color: #718096; margin-bottom: 2rem;'>DenseNet Encoder + LSTM Decoder architecture trained on Flickr8k.</p>", unsafe_allow_html=True)


tab_app, tab_metrics, tab_docs = st.tabs(["Application", "Methodology & Metrics", "Resources"])

# -----------------------------------------------------------------------------
# TAB 1: APPLICATION
# -----------------------------------------------------------------------------
with tab_app:
    col_input, col_display = st.columns([1, 2], gap="large")
    
    image_source = None
    
    with col_input:
        st.subheader("Input")
        st.markdown("<br>", unsafe_allow_html=True) 
        
        
        input_type = st.radio("Source", ["Upload File", "Sample Database"], horizontal=True)
        
        if input_type == "Upload File":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uploaded_file:
                image_source = Image.open(uploaded_file)
        else:
            if os.path.exists(FLICKR_FOLDER):
                images_list = [f for f in os.listdir(FLICKR_FOLDER) if f.endswith(('.jpg', '.png'))]
                selected_img = st.selectbox("Select Sample", images_list, label_visibility="collapsed")
                if selected_img:
                    image_source = Image.open(os.path.join(FLICKR_FOLDER, selected_img))

        if image_source and st.button("Run Inference", use_container_width=True):
            if caption_model:
                with st.spinner("Processing..."):
                    st.session_state['generated_caption'] = predict_caption(caption_model, tokenizer, feature_extractor, image_source)
            else:
                st.error("Model not found.")

    with col_display:
        st.subheader("Result")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if image_source:
            
            st.image(image_source, use_container_width=True)
            
            
            if 'generated_caption' in st.session_state:
                st.markdown(f"""
                <div class="caption-text">
                    {st.session_state['generated_caption']}
                </div>
                """, unsafe_allow_html=True)
        else:
            
            st.markdown("""
            <div style="height: 400px; background-color: #1A202C; border: 1px dashed #2D3748; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #4A5568;">
                No image selected
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: METHODOLOGY & METRICS
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown("<br>", unsafe_allow_html=True)
    
    
    nav_col, content_col = st.columns([1, 4])
    
    with nav_col:
        st.markdown("**Navigation**")
        section = st.radio("Go to", ["Architecture", "Training Loss", "BLEU Analysis"], label_visibility="collapsed")
    
    with content_col:
        
        if section == "Architecture":
            st.markdown("### Model Architecture")
            st.write("The system utilizes a Merge Architecture strategy, injecting visual features at every time step of the sequence generation.")
            
            st.markdown("---")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.graphviz_chart("""
                    digraph {
                        rankdir=LR;
                        bgcolor="#0F1116";
                        node [shape=box, style=filled, fillcolor="#1A202C", fontcolor="#E2E8F0", color="#2D3748", fontname="Inter"];
                        edge [color="#4A5568"];

                        img [label="Image Input", fillcolor="#2B6CB0", fontcolor="white"];
                        cnn [label="DenseNet201\n(Frozen)"];
                        rep [label="RepeatVector\n(37 steps)"];
                        
                        txt [label="Text Input", fillcolor="#2B6CB0", fontcolor="white"];
                        emb [label="Embedding\n(256 dim)"];
                        
                        concat [label="Concatenate", shape=circle, width=0.8];
                        lstm [label="LSTM\n(256 units)"];
                        out [label="Softmax"];

                        img -> cnn -> rep -> concat;
                        txt -> emb -> concat;
                        concat -> lstm -> out;
                    }
                """)
            with c2:
                st.markdown("**Technical Specifications**")
                st.markdown("""
                - **CNN:** DenseNet201 (ImageNet weights)
                - **Feature Vector:** 1920 dimensions
                - **RNN:** Single layer LSTM (256 units)
                - **Embedding Dimension:** 256  
                - **Loss:** Categorical Crossentropy
                """)

        elif section == "Training Loss":
            st.markdown("### Training Convergence")
            st.write("Loss minimization over 50 epochs.")
            
            loss_data = pd.DataFrame({
                'Epoch': range(1, 51),
                'Loss': [5.24, 4.13, 3.71, 3.50, 3.35, 3.23, 3.14, 3.06, 2.99, 2.94, 2.89, 2.85, 2.81, 2.77, 2.74, 2.71, 2.68, 2.66, 2.63, 2.61, 2.59, 2.57, 2.55, 2.54, 2.52, 2.51, 2.49, 2.48, 2.46, 2.45, 2.44, 2.43, 2.42, 2.41, 2.40, 2.39, 2.38, 2.36, 2.35, 2.35, 2.34, 2.33, 2.32, 2.31, 2.30, 2.30, 2.29, 2.28, 2.27, 2.27]
            })
            
            fig = px.line(loss_data, x='Epoch', y='Loss')
            fig.update_layout(
                plot_bgcolor='#0F1116',
                paper_bgcolor='#0F1116',
                font_color='#A0AEC0',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='#2D3748', zeroline=False),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            fig.update_traces(line_color='#63B3ED', line_width=2)
            st.plotly_chart(fig, use_container_width=True)

        #BLEU
        elif section == "BLEU Analysis":
            st.markdown("### Quantitative Evaluation")
            st.write("Performance evaluation on unseen test data using Bilingual Evaluation Understudy (BLEU) scores.")
            
            st.latex(r"BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)")
            
            st.markdown("#### Test Samples")
            st.markdown("---")


            examples = [
                {
                    "img": "assets/flickr8k_images/110595925_f3395c8bd6.jpg", 
                    "ref": "A man in aerodynamic gear riding a bicycle down a road",
                    "pred": "A man rides a bike on a dirt road",
                    "score": 0.76
                },
                {
                    "img": "assets/flickr8k_images/19212715_20476497a3.jpg", 
                    "ref": "A person kayaking in the middle of the ocean",
                    "pred": "A man paddles a canoe on the ocean",
                    "score": 0.58
                },
                {
                    "img": "assets/flickr8k_images/97406261_5eea044056.jpg", 
                    "ref": "A dog and cat are fighting on a chair",
                    "pred": "A dog is biting a cat in the kitchen",
                    "score": 0.42
                },
                {
                    "img": "assets/flickr8k_images/27782020_4dab210360.jpg", 
                    "ref": "People walk around a mobile puppet theater",
                    "pred": "A man in a red jacket is riding a bicycle",
                    "score": 0.15
                }
            ]

 
            for ex in examples:
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 0.5])
                    
                    with c1:
                        if os.path.exists(ex["img"]):
                            st.image(ex["img"], use_container_width=True)
                        else:
                            st.markdown("`Image not found`")
                    
                    with c2:
                        st.markdown(f"**Reference:** <span style='color:#A0AEC0'>{ex['ref']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Prediction:** <span style='color:#E2E8F0'>{ex['pred']}</span>", unsafe_allow_html=True)
                    
                    with c3:
                        color = "#48BB78" if ex['score'] > 0.6 else "#ECC94B" if ex['score'] > 0.4 else "#F56565"
                        st.markdown(f"""
                        <div style="text-align: right;">
                            <span style="font-size: 1.5rem; font-weight: bold; color: {color}">{ex['score']}</span>
                            <br><span style="font-size: 0.8rem; color: #718096">BLEU-1</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()

# -----------------------------------------------------------------------------
# TAB 3: RESOURCES
# -----------------------------------------------------------------------------
with tab_docs:
    st.markdown("### Project Documentation")
    st.write("Details and Reports related to the Image Captioning project.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # 1. GITHUB
    with col1:
        st.markdown("""
        <div class="css-card">
            <span style="font-size: 2rem;">💻</span>
            <div style="flex-grow: 1;">
                <h4 style="margin-top: 10px;">Source Code</h4>
                <p style="color: #A0AEC0; font-size: 0.9rem;">
                    Github Code with the model classes and with the main train.py file
                </p>
            </div>
            <a href="https://github.com/abd2OO4/image-captioning.git" target="_blank" style="text-decoration: none;">
                <div style="background-color: #2D3748; color: white; padding: 10px; border-radius: 4px; text-align: center; margin-top: 15px; font-weight: 500;">
                    Visit Repository →
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. SLIDES
    with col2:

        slides_url = "https://drive.google.com/file/d/1Hp4CQxDShfxH7L5B2U6LA4mKaalmZ7R1/view?usp=sharing"
        
        st.markdown(f"""
        <div class="css-card">
            <span style="font-size: 2rem;">🎬</span>
            <div style="flex-grow: 1;">
                <h4 style="margin-top: 10px;">Behind the Scenes</h4>
                <p style="color: #A0AEC0; font-size: 0.9rem;">
                    Presentation slides for easier understanding
                </p>
            </div>
            <a href="{slides_url}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #2B6CB0; color: white; padding: 10px; border-radius: 4px; text-align: center; margin-top: 15px; font-weight: 500;">
                    View Presentation →
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)

    # 3. REPORT
    with col3:
        report_url = "https://drive.google.com/file/d/1QEAAMEnOw8149xNZhPaQbGoVz8JFGvkd/view?usp=sharing"
        
        st.markdown(f"""
        <div class="css-card">
            <span style="font-size: 2rem;">📄</span>
            <div style="flex-grow: 1;">
                <h4 style="margin-top: 10px;">Final Report</h4>
                <p style="color: #A0AEC0; font-size: 0.9rem;">
                    Final proj Report with literature review
                </p>
            </div>
            <a href="{report_url}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #38A169; color: white; padding: 10px; border-radius: 4px; text-align: center; margin-top: 15px; font-weight: 500;">
                    Read Report →
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Summary (Keras)")
    with st.expander("Expand details"):
        st.code("""
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
image_input (InputLayer)        (None, 1920)         0                                            
dense (Dense)                   (None, 256)          491,776     image_input[0][0]                
text_input (InputLayer)         (None, 37)           0                                            
embedding (Embedding)           (None, 37, 256)      2,247,936   text_input[0][0]                 
concatenate (Concatenate)       (None, 37, 512)      0           repeat_vector[0][0]              
lstm (LSTM)                     (None, 256)          787,456     concatenate[0][0]                
output (Dense)                  (None, 8781)         1,132,749   dense_1[0][0]                    
==================================================================================================
Total params: 4,692,813
Trainable params: 4,692,813
        """)

    