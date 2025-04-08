# Multi-Model Retrival System
We're building an intelligent product search backend using FastAPI, CLIP, and FAISS. The system helps users:
- Search for fashion products using an image, text, or both.
- Find visually and semantically similar items from a fashion dataset.
- Get quick and accurate search results powered by precomputed embeddings and FAISS indexing.

## CLIP Model 
- Uses OpenAI’s CLIP (ViT-B/32) model to convert both images and text into embeddings in the same vector space.
- These embeddings capture visual + textual meaning of fashion items.

## Embedding Store + FAISS Index
- All products in the dataset are preprocessed:
  - Image and caption embeddings are averaged and saved.
- A FAISS index is built using these embeddings for fast similarity search using cosine distance.

## FastAPI Backend
Provides a /upload endpoint to receive:
- An image (img)

- A text caption (text)

- Optional filter like color (can be extended)
Computes the embedding of the query input.
Uses FAISS to find top-5 similar products from the dataset.

# Dataset
We're using the `ashraq/fashion-product-images-small` dataset from HuggingFace.
