import cv2
import numpy as np
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List

model = YOLO("chess_model/best.pt")

fen_map = {
    "white-king": "K", "black-king": "k",
    "white-queen": "Q", "black-queen": "q",
    "white-rook": "R", "black-rook": "r",
    "white-bishop": "B", "black-bishop": "b",
    "white-knight": "N", "black-knight": "n",
    "white-pawn": "P", "black-pawn": "p",
}

class Detection(BaseModel):
    class_: str
    bbox: List[float]

def detect_board_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=50)
    if corners is None or len(corners) < 4:
        return None
    corners = np.int0(corners).reshape(-1, 2)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = corners[np.argmin(s)]       # top-left
    rect[2] = corners[np.argmax(s)]       # bottom-right
    rect[1] = corners[np.argmin(diff)]    # top-right
    rect[3] = corners[np.argmax(diff)]    # bottom-left
    return rect

def warp_board(image, corners, size=416):
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (size, size))

def filter_detections(results, square_size=52):
    filtered = {}
    if results.boxes is None:
        return filtered

    boxes = results.boxes.xywh.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x, y, _, _ = box
        col = int(x // square_size)
        row = int(y // square_size)
        key = (row, col)

        if key not in filtered or conf > filtered[key][2]:
            filtered[key] = (cls_id, x, conf)
    return filtered

def detections_to_fen(boxes, names, image_size=416, frame_width=416, frame_height=416):
    board = [["" for _ in range(8)] for _ in range(8)]
    square_w = frame_width / 8
    square_h = frame_height / 8

    if boxes is None or boxes.data is None:
        return "8/8/8/8/8/8/8/8 w KQkq - 0 1"

    for box, cls_id in zip(boxes.xywh, boxes.cls):
        x_center, y_center, w, h = box.tolist()

        col = int(x_center // square_w)
        row = int(y_center // square_h)

        # Clamp to board
        col = min(max(col, 0), 7)
        row = min(max(row, 0), 7)

        # row = 7 - row  

        class_name = names[int(cls_id.item())]
        piece = fen_map.get(class_name)
        if piece:
            board[row][col] = piece

    fen_rows = []
    for row in board:
        fen_row = ""
        empty = 0
        for square in row:
            if square == "":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += square
        if empty:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    return f"{'/'.join(fen_rows)} w KQkq - 0 1"

def detect_and_annotate(frame, return_fen=False):
    results = model(frame)[0]
    
    annotated_frame = frame.copy()
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fen = ""
    if return_fen:
        height, width = frame.shape[:2]
        fen = detections_to_fen(results.boxes, results.names, frame_width=width, frame_height=height)

    return annotated_frame, fen

