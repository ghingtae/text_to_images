# ----------------
# app.py
# ----------------

import streamlit as st
from PIL import Image
import numpy as np
import mlx.core as mx
from stable_diffusion import StableDiffusion

def generate_images(prompt, n_images=4, steps=50, cfg=7.5, negative_prompt="", n_rows=2):
    sd = StableDiffusion()

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        negative_text=negative_prompt,
    )
    # Edited Feb 20 2024 - removes mx.simplify(x_t) [https://github.com/ml-explore/mlx-examples/pull/379]
    for x_t in latents:
        mx.eval(x_t)

    # Decode them into images
    decoded = []
    for i in range(0, n_images):
        decoded_img = sd.decode(x_t[i:i+1])
        mx.eval(decoded_img)
        decoded.append(decoded_img)

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Convert to PIL Image
    # Edited Feb 20 2024 to 
    return Image.fromarray(np.array(x))

st.set_page_config(page_title="Stable Diffusion Image Generator")
st.title("Stable Diffusion Image Generator")
st.write("Generate images from a textual prompt using Stable Diffusion")

prompt = st.text_input("Prompt", "")
n_images = st.slider("Number of Images", 1, 10, 4)
steps = st.slider("Steps", 20, 150, 50, help="The number of steps the diffusion model will repeat to turn random noise into a recognizable image. Typically between 50 and 150.")
cfg = st.slider("CFG Weight", 0.0, 20.0, 7.5, help="The CFG Weight controls how much the input prompt or image influences the generated image. Typically between 7.0 and 13.0.")
negative_prompt = st.text_input("Negative Prompt", "")
n_rows = st.slider("Number of Rows", 1, 10, 1)

if st.button("Generate Images"):
    images = generate_images(prompt, n_images, steps, cfg, negative_prompt, n_rows)
    st.image(images, use_column_width=True)
