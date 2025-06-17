import cv2
import numpy as np

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
    h, w = img.shape[:2]

    # 1. Preprocessing: Grayscale & Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thresholding (lebih stabil dari Canny untuk kasus ini)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # 3. Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return get_default_board_corners(img)

    # 4. Filter kontur kecil
    min_area = h * w * 0.2  # hanya kontur besar
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not contours:
        print("default border")
        return get_default_board_corners(img)

    # 5. Ambil kontur terbesar
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. Gunakan bounding box dari kontur terbesar
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
    return [
        [x, y],
        [x + w_box, y],
        [x + w_box, y + h_box],
        [x, y + h_box]
    ]
