from ultralytics import YOLO
from utils import warp_image, get_board_corners
import numpy as np
import cv2

model = YOLO("chess_model/best.pt")

fen_map = {
    "white-king": "K", "black-king": "k",
    "white-queen": "Q", "black-queen": "q",
    "white-rook": "R", "black-rook": "r",
    "white-bishop": "B", "black-bishop": "b",
    "white-knight": "N", "black-knight": "n",
    "white-pawn": "P", "black-pawn": "p",
}

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


def build_fen(filtered_dets):
    board = [["" for _ in range(8)] for _ in range(8)]
    for (row, col), (cls_id, _, _) in filtered_dets.items():
        label = model.names[cls_id]
        piece = fen_map.get(label, "")
        if piece:
            board[row][col] = piece

    fen_rows = []
    for row in board:
        fen_row = ""
        empty = 0
        for sq in row:
            if sq == "":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += sq
        if empty:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w KQkq - 0 1"

def detect_fen_from_image(pil_image):
    img = np.array(pil_image)
    corners = get_board_corners(img) 
    warped = warp_image(img, corners)
    results = model(warped)[0]
    filtered = filter_detections(results)
    return build_fen(filtered)
