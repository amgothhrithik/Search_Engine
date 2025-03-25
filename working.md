#### Goal: Given a query  for example *'A stylish red shoes'* or a image and other parameters, filter and give relavent products.
- We will use clip-based model as it can find both embedding vector for both image and text.

## Clip Models(by OPENAI)
- CLIP's are multi-modal AI models that understands both images and texts.
- These are trained using `'Contrastive learning'` approach.
   - In this approach, model are trained to distinguish similar and dissimliar text-image pairs by pulling similar pair embedding closer and by pushing dissimilar pairs emdedding farther apart.
     - Positive Pairs: an image of dog and a text describing dog.
     - Negitive Pairs: An image of a dog and a caption "A red Apple".
  - Uses Cosine similarity to measure the similarity between text and image embeddings.
  -  Enables efficient image-text search applications.
- Cosine Similarity- It is the angle b/w text and image embedding.
  - They are scale-invariant, as there are normalized and do to this cosine similarity rarely goes negitive.
- Clip model contain 2 encoders
  1. Text Encoder(like a Transformer) processes input text.
  2. Vision Encoder(like a Vision Transformer or ResNet) processes input images.
- No Need for Labeled Data(Zero-Shot Learning).

## CLIPProcessor
- text: Converts text into tokenizers, so that the model understand.
- Image:
  - Reshapes images to a fixed shape(224×224).
  - Converts into a tensor.
  - Normalizes the pixel values.
- Ensures consistency
- Makes it easy to run CLIP models without worrying about data formatting.
- Output of CLIPProceeser looks like:
  - {'`input_ids`':  , '`attention_mask`':   , '`pixel_values`':  }
  - 'input_ids' are the tokenized text.
  - 'attention_mask' are used to ignore the padded tokens in the 'input_ids' during computation.
    - 1 → real tokens.
    - 0 → padded tokens.
  - shape of 'input_ids' -(batch_size, max_seq_length),
  - shape of 'attention_mask'-(batch_size, max_seq_length)
  - shape of 'pixel_values'-[batch_size, 3, 224,224]
 
## 
- we are using `"openai/clip-vit-base-patch32"` model for CLIPProcessor, CLIPModel
- we are using `'productDisplayName'` column for computing text embedding from dataset and `'image'` column which contain actual image in size 60x80 pixel for finding image embedding.
- we find the aver. embedding of text and image using  `0.5 * text_embeddings + 0.5 * image_embeddings` as it captures both text and image information.
- we store aver. embedding for each product in numpy array.
- These aver. embeddings are used for filtering the products based on query embedding.


## FAISS(Facebook AI Similarity Search)
- It is a open source library for efficient similarity search and clustering of dense vectors.
- Using `faiss.normalize_L2(embeddings_np)`, we normailze the embeddings along last dim(1).
- Why normalization?
  - FAISS uses dot product for similarity search.If normalized, dor product = Cosine similarity.
- Since we have small dataset of 44k, we are using `faiss.IndexFlatIP(512)`.

### faiss.IndexFlatIP(512)
- In this, there is no pre-processing or clustering.
- All vectors are stored in memory.
- Query vector's dop product is computed for all vectors.
#### index.search(text_embedding.cpu().numpy(), 5)
- It returns top-5 nearest neighbors with distance and indices.

### In  IndexIVFFlat and IndexIVFPQ
- With large dataset, we will first cluster the datapoint and find k-top datapoint from that clusters.

### faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
- A multi-layered graph is constructed, where each node is vector and is connected to atmost M nearest neighbours.
- Use M ≈ log(N)
- Searching:
   - Starts from entry points.
   - Using a greedy search, we move closer to query vector and expands the search to neighbours.
   - In the end return top-k most similar vectors.

