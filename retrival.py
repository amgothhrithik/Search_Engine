import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_dataset
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")

import pandas as pd
df = pd.DataFrame(dataset)

from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



import faiss
import numpy as np
path = r"D:\env\search_engine\embeddings.npy"

embeddings = np.load(path)

# Convert PyTorch tensor to NumPy (FAISS works with NumPy)
embeddings_np = embeddings.astype("float32")
faiss.normalize_L2(embeddings_np)
index = faiss.IndexFlatIP(512)  # Inner Product (Cosine when normalized)
index.add(embeddings_np)


