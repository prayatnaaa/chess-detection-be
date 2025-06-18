from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import cv2
import numpy as np
from sse_starlette import EventSourceResponse
from detector import detect_and_annotate
from typing import List
import io
from PIL import Image
from ultralytics import YOLO
from detector import detections_to_fen
import asyncio
from uuid import uuid4

app = FastAPI()
model = YOLO("chess_model/best.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

fen_history: List[str] = []

def read_image_file(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.get("/stream-fen")
async def stream_fen(session_id: str):
    video_path = video_sessions.get(session_id)
    if not video_path:
        return JSONResponse(status_code=404, content={"error": "Invalid session ID"})

    async def event_generator():
        cap = cv2.VideoCapture(video_path)
        last_fen = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)[0]
            if results.boxes is None or len(results.boxes) == 0:
                await asyncio.sleep(0.01)
                continue

            fen = detections_to_fen(results.boxes, results.names, image_size=frame.shape[0])
            if fen != last_fen:
                fen_history.append(fen)
                last_fen = fen
                yield {"event": "fen_update", "data": fen}

            await asyncio.sleep(0.01)

        cap.release()
        yield {"event": "end", "data": "done"}

    return EventSourceResponse(event_generator())


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
    contents = file.file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    annotated, fen = detect_and_annotate(frame, return_fen=True)
    return {"fen": fen}

video_sessions = {}  # store session_id -> video path

@app.post("/upload-video-stream")
async def upload_video_stream(file: UploadFile = File(...)):
    session_id = str(uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    video_sessions[session_id] = video_path
    return {"session_id": session_id}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)

    last_fen = None
    detected_fens = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        fen = detections_to_fen(results.boxes, results.names, image_size=frame.shape[0])
        if fen != last_fen:
            fen_history.append(fen)

        # Detect FEN change (new board state)
        if fen != last_fen:
            detected_fens.append(fen)
            last_fen = fen

    cap.release()
    return {"fens": detected_fens}

@app.get("/fen-history")
def get_fen_history():
    return JSONResponse(content={"history": fen_history})

@app.get("/reset-history")
def reset_history():
    fen_history.clear()
    return {"message": "History cleared."}
