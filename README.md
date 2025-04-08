# Multi-Model Retrival System
We're building an intelligent product search backend using FastAPI, CLIP, and FAISS. The system helps users:
- Search for fashion products using an image, text, or both.
- Find visually and semantically similar items from a fashion dataset.
- Get quick and accurate search results powered by precomputed embeddings and FAISS indexing.

## Architecture Overview
Components:

1. CLIP Model:
   - Model: openai/clip-vit-base-patch32
   - Converts images and text into 512-dimensional vector embeddings
   - Combines both image and text embeddings by averaging
2. Embedding + FAISS Indexing:
   - All dataset items are embedded and saved as .npy file
   - Embeddings are L2-normalized
   - Stored in a FAISS index (inner product → cosine similarity)

3. Search API (FastAPI):
   - POST /upload: Accepts image, text, and optional color.
   - Computes embedding for the query.
   - Searches FAISS index for top-5 similar items.
   - Returns image results encoded in base64 format
  


# Dataset
We're using the `ashraq/fashion-product-images-small` dataset from HuggingFace.
