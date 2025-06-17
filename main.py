from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from detector import detect_and_annotate
from typing import List
import io
from PIL import Image
from ultralytics import YOLO
from detector import detections_to_fen

app = FastAPI()
model = YOLO("chess_model/best.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fen_history: List[str] = []

def read_image_file(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.post("/detect-fen/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(img)
    result = results[0]

    # Generate FEN
    fen = detections_to_fen(result.boxes, result.names, img.shape[0])

    return JSONResponse(content={"fen": fen})

@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):
    frame = read_image_file(file)
    annotated, fen = detect_and_annotate(frame, return_fen=True)

    if not fen_history or fen != fen_history[-1]:
        fen_history.append(fen)

    _, img_bytes = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(img_bytes.tobytes()), media_type="image/jpeg")

@app.post("/upload-video")
def upload_video(file: UploadFile = File(...)):
    contents = file.file.read()
    np_video = np.frombuffer(contents, np.uint8)
    video = cv2.imdecode(np_video, cv2.IMREAD_COLOR)

    # Optional: Write to file or loop over frames if decoding video stream
    annotated, fen = detect_and_annotate(video, return_fen=True)
    if not fen_history or fen != fen_history[-1]:
        fen_history.append(fen)

    _, img_bytes = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(img_bytes.tobytes()), media_type="image/jpeg")

@app.get("/fen-history")
def get_fen_history():
    return JSONResponse(content={"history": fen_history})

@app.get("/reset-history")
def reset_history():
    fen_history.clear()
    return {"message": "History cleared."}
