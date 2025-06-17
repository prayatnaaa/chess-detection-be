import cv2
import numpy as np
from ultralytics import YOLO

def warp_image(img, src_points, dst_size=416):
    dst_points = np.float32([
        [0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]
    ])
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
    warped = cv2.warpPerspective(img, matrix, (dst_size, dst_size))
    return warped

def get_default_board_corners(img):
    h, w = img.shape[:2]
    return [
        [w * 0.1, h * 0.1],
        [w * 0.9, h * 0.1],
        [w * 0.9, h * 0.9],
        [w * 0.1, h * 0.9]
    ]

def get_board_corners(img):
    # Gunakan YOLO untuk deteksi papan
    yolo_model = YOLO("chess_model/best.pt")
    result = yolo_model(img)[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        if label == "board":  # pastikan label ini sesuai label di model
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]
    
    # fallback jika papan tidak terdeteksi
    print("Default board fallback")
    return get_default_board_corners(img)

