import streamlit as st
import os
import zipfile
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import faiss

# --- Setup Persistent Directories ---
IMAGE_DIR = "./uploaded_images"
FAISS_DIR = "./faiss_index"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(FAISS_DIR, "image_index.faiss")
NAMES_PATH = os.path.join(FAISS_DIR, "image_names.npy")

# --- Load CLIP Model ---
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

model, processor, device = load_clip_model()

# --- Embedding Functions ---
def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# --- Build FAISS Index ---
def build_faiss_index(image_folder: str):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        st.warning("⚠️ No images found in folder!")
        return 0

    embeddings = []
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        emb = get_image_embedding(img)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    np.save(NAMES_PATH, np.array(image_files))
    st.success(f"✅ FAISS index built successfully with {len(image_files)} images!")
    return len(image_files)

# --- Search Function ---
def search_similar(query_emb, top_k=5):
    if not os.path.exists(INDEX_PATH):
        st.error("⚠️ FAISS index not found! Build it first.")
        return []

    index = faiss.read_index(INDEX_PATH)
    image_files = np.load(NAMES_PATH)
    D, I = index.search(query_emb, top_k)
    return [(image_files[i], D[0][j]) for j, i in enumerate(I[0])]

# --- Streamlit UI ---
st.title("🔍 Image & Text Similarity Search (CLIP + FAISS)")

st.markdown("""
Upload your image dataset (multiple files or a ZIP folder), 
build embeddings with CLIP, and search similar images using text or image queries.
""")

# --- Upload Options ---
upload_type = st.radio("Choose upload type:", ("Upload Multiple Images", "Upload ZIP File"))

if upload_type == "Upload Multiple Images":
    uploaded_images = st.file_uploader("📁 Upload multiple image files", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded_images:
        for uploaded_file in uploaded_images:
            img_path = os.path.join(IMAGE_DIR, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_images)} images uploaded successfully!")
        if st.button("🧠 Build Embedding Index"):
            build_faiss_index(IMAGE_DIR)

elif upload_type == "Upload ZIP File":
    uploaded_zip = st.file_uploader("📦 Upload ZIP file containing images", type=["zip"])
    if uploaded_zip:
        zip_path = os.path.join(IMAGE_DIR, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(IMAGE_DIR)
        os.remove(zip_path)
        st.success("✅ ZIP extracted successfully!")
        if st.button("🧠 Build Embedding Index"):
            build_faiss_index(IMAGE_DIR)

# --- Query Section ---
st.subheader("🔎 Search Using Text or Image")

query_type = st.radio("Select query type:", ("Text", "Image"))
query_emb = None

if query_type == "Text":
    query_text = st.text_input("Enter your text query:")
    if query_text:
        query_emb = get_text_embedding(query_text)

elif query_type == "Image":
    query_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Image", use_container_width=True)
        query_emb = get_image_embedding(query_img)

# --- Search Results ---
if query_emb is not None:
    with st.spinner("🔍 Searching for similar images..."):
        results = search_similar(query_emb, top_k=5)

    if results:
        st.subheader("Top Similar Images:")
        cols = st.columns(5)
        for i, (img_name, dist) in enumerate(results):
            img_path = os.path.join(IMAGE_DIR, img_name)
            if os.path.exists(img_path):
                with cols[i % 5]:
                    st.image(img_path, caption=f"{img_name}\n(dist={dist:.4f})", use_container_width=True)
