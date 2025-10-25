import streamlit as st
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from PIL import Image

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="Sketch to Real Image Generator", layout="centered")
st.title("✏️ Sketch to Realistic Image Generator")
st.write("Upload a hand-drawn sketch and generate a realistic version using ControlNet (Scribble).")

# File uploader
uploaded_file = st.file_uploader("Upload your sketch image (PNG or JPG):", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Enter a description (prompt):", "a realistic photo of an airplane flying in the sky")

generate_button = st.button("Generate Realistic Image")

# ---------------------------------------
# Model loading
# ---------------------------------------
@st.cache_resource
def load_pipeline():
    st.info("Loading ControlNet + Stable Diffusion model... Please wait.")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
    )
    pipe.to("cpu")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.to("cpu")
    return pipe

pipe = load_pipeline()

# ---------------------------------------
# Generate image
# ---------------------------------------
if uploaded_file and generate_button:
    st.image(uploaded_file, caption="Uploaded Sketch", use_column_width=True)
    sketch = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Generating realistic image..."):
        result = pipe(prompt, image=sketch, num_inference_steps=30)
        generated_image = result.images[0]
        st.image(generated_image, caption="Generated Realistic Image", use_column_width=True)
        generated_image.save("output.png")

        with open("output.png", "rb") as file:
            st.download_button(label="Download Image", data=file, file_name="realistic_output.png", mime="image/png")

st.markdown("---")
st.caption("Built with ❤️ using Stable Diffusion + ControlNet")
