from fastapi import FastAPI,UploadFile,File,Form
from typing import Optional
from fastapi.responses import JSONResponse
import uvicorn
from retrival import model,processor,index,device,df,torch,faiss
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from typing import List
app = FastAPI(docs_url = "/docs", redoc_url = None)
# @app.post("/debug")
# async def debug(img: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), color: Optional[str] = Form(None)):
#     return {
#         "img": img.filename if img else None,
#         "text": text,
#         "color": color
#     }

@app.post("/upload")
async def upload(
    img: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    color: Optional[str] = Form(None)
):
    print(f"text={text}, color={color}, img={img.filename if img else 'None'}")
    return await process_input(img, text, color)  # <-- Pass the actual UploadFile object


async def process_input(img,text,color):
    img_array = None
    if img:

        contents = await img.read()
        img_array = np.array(Image.open(BytesIO(contents)).convert("RGB"))



    if img_array is None and text is None:
        return {"error": "No input given"}
    query_embedding=encode(img_array,text)
    faiss.normalize_L2(query_embedding.numpy())
    D, I = index.search(query_embedding, 5)
    base64_images: List[str] = []
    for idx in I[0]:
        img = df.loc[idx, "image"]  # PIL image directly
        base64_str = pil_to_base64(img)
        base64_images.append(base64_str)

    return JSONResponse(content={"images": base64_images})


def encode(img_array=None, text=None):
    kwargs = {"return_tensors": "pt"}
    if img_array is not None:
        kwargs["images"] = img_array
    if text is not None:
        kwargs["text"] = text
        kwargs["padding"] = True

    inputs = processor(**kwargs).to(device)

    with torch.no_grad():
        if img_array is not None and text is not None:
            outputs = model(**inputs)
            return 0.5 * outputs.image_embeds.cpu() + 0.5 * outputs.text_embeds.cpu()
        elif img_array is not None:
            img_embed = model.get_image_features(**inputs)
            return img_embed.cpu()
        else:
            txt_embed = model.get_text_features(**inputs)
            return txt_embed.cpu()
def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

if __name__ == "__main__":
    uvicorn.run("api_connector:app", host="localhost", port=5000, reload=True)

