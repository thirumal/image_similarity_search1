import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import streamlit as st
from urllib.parse import urlparse
import requests
from io import BytesIO

# --- Paths ---
IMAGE_DIR = "./images"
FAISS_DIR = "./faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

INDEX_PATH = os.path.join(FAISS_DIR, "image_index.faiss")
NAMES_PATH = os.path.join(FAISS_DIR, "image_names.npy")

# --- Load CLIP Model ---
@st.cache_resource
def load_clip_model():
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN")
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
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
        st.warning("No images found in folder!")
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
    return len(image_files)

# --- Search Function ---
def search_similar(query_emb, top_k=5):
    if not os.path.exists(INDEX_PATH):
        st.error("FAISS index not found! Build it first.")
        return []

    index = faiss.read_index(INDEX_PATH)
    image_files = np.load(NAMES_PATH)
    D, I = index.search(query_emb, top_k)
    return [(image_files[i], D[0][j]) for j, i in enumerate(I[0])]

# --- Streamlit UI ---
st.title("🔍 Image/Text Similarity Search")

# Option: Local folder or S3
source_option = st.radio("Select image source:", ("Local folder", "S3 Bucket"))

if source_option == "Local folder":
    folder_path = st.text_input("Enter image folder path:", IMAGE_DIR)
elif source_option == "S3 Bucket":
    s3_url = st.text_input("Enter S3 bucket URL (with folder path):")
    folder_path = IMAGE_DIR
    if s3_url:
        os.makedirs(folder_path, exist_ok=True)
        st.info("Downloading images from S3...")
        parsed_url = urlparse(s3_url)
        # This is a simple example; for actual S3, you might use boto3
        # Here we assume s3_url is a public URL to a folder containing images
        response = requests.get(s3_url)
        # You can enhance to download all images in folder

# Build/update FAISS index
if st.button("📦 Build/Update FAISS Index"):
    count = build_faiss_index(folder_path)
    st.success(f"FAISS index built successfully with {count} images!")

# Choose query type
query_type = st.radio("Query type:", ("Image", "Text"))
query_emb = None

if query_type == "Image":
    uploaded_file = st.file_uploader("Upload a query image:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Query Image", use_container_width=True)
        query_emb = get_image_embedding(query_img)

elif query_type == "Text":
    query_text = st.text_input("Enter your text query:")
    if query_text:
        query_emb = get_text_embedding(query_text)

# Search and display results
if query_emb is not None:
    with st.spinner("Searching for similar images..."):
        results = search_similar(query_emb, top_k=5)

    if results:
        st.subheader("Top Similar Images:")
        cols = st.columns(5)
        for i, (img_name, dist) in enumerate(results):
            img_path = os.path.join(folder_path, img_name)
            if os.path.exists(img_path):
                with cols[i % 5]:
                    st.image(img_path, caption=f"{img_name}\n(dist={dist:.4f})", use_container_width=True)
