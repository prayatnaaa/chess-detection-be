from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

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

class DetectionsRequest(BaseModel):
    detections: List[Detection]

def detections_to_fen(detections, image_size=416):
    board = [["" for _ in range(8)] for _ in range(8)]
    square_size = image_size / 8

    for det in detections:
        cls = det.class_
        x, y, w, h = det.bbox

        col = int(x // square_size)
        row = int(y // square_size)

        board_row = row
        board_col = col

        piece = fen_map.get(cls)
        if piece:
            board[board_row][board_col] = piece

    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for square in row:
            if square == "":
                empty_count += 1
            else:
                if empty_count:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    board_fen = "/".join(fen_rows)
    return f"{board_fen} w KQkq - 0 1"

@app.post("/generate-fen")
def generate_fen(request: DetectionsRequest):
    fen = detections_to_fen(request.detections)
    return {"fen": fen}
