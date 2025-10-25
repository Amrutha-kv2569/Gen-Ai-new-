import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(page_title="Sketch to Image Generator", page_icon="🎨", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {width: 100%; background-color: #4CAF50; color: white; height: 3em; border-radius: 10px; font-weight: bold;}
    .stButton>button:hover {background-color: #45a049;}
    h1 {color: #2c3e50;}
    h3 {color: #34495e;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Sketch-to-Image Generator")
st.markdown("### Transform your sketches into realistic images using Generative AI")
st.markdown("---")

@st.cache_resource
def load_generator_model():
    model_path = "models/generator_model.h5"
    
    # If model doesn't exist locally, download from Google Drive
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        
        # Replace with your Google Drive file ID
        file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # ⚠️ CHANGE THIS
        
        if file_id == "https://drive.google.com/file/d/1U-AP7epB0aY_zDw_T1aQArcAJtINw0W4/view?usp=sharing":
            st.error("⚠️ Please update the Google Drive file ID in app.py")
            return None
        
        try:
            import gdown
            st.info("📥 Downloading model from Google Drive (first time only, ~2 minutes)...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.error("Please check the Google Drive file ID and permissions.")
            return None
    
    # Load the model
    try:
        model = keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_sketch(image, target_size=(256, 256)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image)
    img_array = (img_array / 127.5) - 1.0
    return img_array

def postprocess_image(generated_image):
    image = (generated_image + 1.0) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def generate_image(model, sketch):
    sketch_batch = np.expand_dims(sketch, axis=0)
    generated = model.predict(sketch_batch, verbose=0)
    generated_image = generated[0]
    return postprocess_image(generated_image)

def main():
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a **Pix2Pix GAN** architecture to convert sketches into realistic images.
        
        **How to use:**
        1. Upload a sketch image
        2. Click 'Generate Image'
        3. View and download the result
        """)
        st.markdown("---")
        st.header("⚙️ Settings")
        img_size = st.selectbox("Image Size", [256, 512], index=0)
        st.markdown("---")
        st.header("📊 Model Info")
        if os.path.exists("models/generator_model.h5"):
            st.success("✅ Model Status: Loaded")
            file_size = os.path.getsize("models/generator_model.h5") / (1024 * 1024)
            st.info(f"Model Size: {file_size:.2f} MB")
        else:
            st.error("❌ Model Status: Not Found")
            st.warning("Please train the model first!")
        st.markdown("---")
        st.header("🎯 Tips")
        st.write("""
        - Use clear, bold sketches
        - Simple line drawings work best
        - Avoid too much detail
        - Try different sketch styles
        """)
    
    generator = load_generator_model()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Sketch")
        uploaded_file = st.file_uploader("Choose a sketch image...", type=['png', 'jpg', 'jpeg'], help="Upload a hand-drawn or digital sketch")
        if uploaded_file is not None:
            sketch_image = Image.open(uploaded_file)
            st.image(sketch_image, caption="Uploaded Sketch", use_container_width=True)
            st.info(f"📐 Original Size: {sketch_image.size[0]} x {sketch_image.size[1]}")
            if st.button("🎨 Generate Image", type="primary"):
                if generator is None:
                    st.error("❌ Model not loaded. Please train the model first.")
                else:
                    with st.spinner("🎨 Generating image... Please wait..."):
                        try:
                            processed_sketch = preprocess_sketch(sketch_image, (img_size, img_size))
                            generated_img = generate_image(generator, processed_sketch)
                            st.session_state['generated_image'] = generated_img
                            st.session_state['sketch_image'] = np.array(sketch_image.resize((img_size, img_size)))
                            st.success("✅ Image generated successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error generating image: {e}")
        else:
            st.info("👆 Please upload a sketch image to get started")
    
    with col2:
        st.subheader("🖼️ Generated Image")
        if 'generated_image' in st.session_state:
            generated_pil = Image.fromarray(st.session_state['generated_image'])
            st.image(generated_pil, caption="Generated Image", use_container_width=True)
            buf = io.BytesIO()
            generated_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            st.download_button(label="⬇️ Download Generated Image", data=byte_im, file_name="generated_image.png", mime="image/png", type="primary")
        else:
            st.info("👈 Upload a sketch and click 'Generate Image' to see results here.")
            placeholder = np.ones((256, 256, 3), dtype=np.uint8) * 240
            st.image(placeholder, caption="Waiting for generation...", use_container_width=True)
    
    if 'generated_image' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Side-by-Side Comparison")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.image(st.session_state['sketch_image'], caption="Input Sketch", use_container_width=True)
        with comp_col2:
            st.image(st.session_state['generated_image'], caption="Generated Image", use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Sample Ideas")
    st.write("Try sketching these objects for best results:")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        st.info("🏠 **Houses**\nSimple building sketches")
    with sample_col2:
        st.info("🐱 **Animals**\nCats, dogs, birds")
    with sample_col3:
        st.info("🚗 **Vehicles**\nCars, bikes, planes")
    with sample_col4:
        st.info("🌳 **Nature**\nTrees, flowers, landscapes")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Pix2Pix GAN Architecture for Sketch-to-Image Generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
