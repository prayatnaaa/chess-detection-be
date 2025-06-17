from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detector import detect_fen_from_image
from PIL import Image
import io

app = FastAPI()

# CORS if frontend is separate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-fen")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    fen = detect_fen_from_image(img)
    return {"fen": fen}
