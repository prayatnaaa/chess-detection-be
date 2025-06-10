import cv2
import numpy as np
from ultralytics import YOLO
import chess
import chess.svg
import cairosvg
import tkinter as tk
import copy

# === Inisialisasi papan dan model ===
board = chess.Board()
last_state = {}
model = YOLO('chess_model/best.pt')  # Ganti path model YOLO kamu
log_moves = []
rejected_moves = set()  # Cache langkah ilegal
stable_counter = 0
required_stable_frames = 3  # Jumlah frame yang harus stabil

# === GUI Tkinter ===
root = tk.Tk()
root.title("Log Langkah Catur")
listbox = tk.Listbox(root, width=30, height=30, font=("Courier", 12))
listbox.pack(padx=10, pady=10)

def update_gui(move_str):
    print(f"Update GUI: {move_str}")
    listbox.insert(tk.END, move_str)
    listbox.see(tk.END)

update_gui("GUI Siap")

# === Fungsi gambar papan ===
def draw_board(board, size=480):
    board_svg = chess.svg.board(board=board, size=size)
    png_bytes = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), output_width=size, output_height=size)
    png_arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(png_arr, cv2.IMREAD_COLOR)
    return img

# === Fungsi posisi ke notasi catur (dengan pembalikan orientasi papan) ===
def to_notation(row, col):
    # Asumsi: kiri atas adalah h1 → horizontal di-flip, vertikal tetap
    file = chr(ord('a') + (7 - col))  # flip horizontal
    rank = str(1 + row)               # no flip vertically
    return file + rank

# === Ambil posisi bidak dari hasil YOLO ===
def get_piece_positions(results):
    piece_pos = {}
    boxes = results[0].boxes
    if boxes is None:
        return piece_pos

    cell_size = 480 // 8

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = boxes.conf[i].cpu().numpy()
        if conf < 0.3:
            continue
        x_center = int((xyxy[0] + xyxy[2]) / 2)
        y_center = int((xyxy[1] + xyxy[3]) / 2)
        col = min(x_center // cell_size, 7)
        row = min(y_center // cell_size, 7)
        notation = to_notation(row, col)
        piece_pos[notation] = True

        # Untuk debug: tampilkan notasi di gambar
        x_disp = col * cell_size + 5
        y_disp = row * cell_size + 20
        cv2.putText(results[0].orig_img, notation, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return piece_pos


# === Deteksi langkah dari perubahan posisi ===
def detect_move(prev, curr):
    removed = [pos for pos in prev if pos not in curr]
    added = [pos for pos in curr if pos not in prev]

    print(f"Removed: {removed}")
    print(f"Added: {added}")

    if len(removed) == 1 and len(added) == 1:
        return removed[0], added[0]
    return None, None

# === Proses video ===
cap = cv2.VideoCapture('Chess_video_example.mp4')  # Ganti dengan path video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (480, 480))
    results = model(frame_resized, verbose=False)
    current_state = get_piece_positions(results)

    # Gambar hasil deteksi kotak
    annotated_frame = frame_resized.copy()
    boxes = results[0].boxes
    if boxes is not None:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = boxes.conf[i].cpu().numpy()
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Stabilitas deteksi
    if current_state == last_state:
        stable_counter += 1
    else:
        stable_counter = 0

    # Jika stabil selama beberapa frame, proses deteksi langkah
    if stable_counter >= required_stable_frames and last_state:
        from_pos, to_pos = detect_move(last_state, current_state)

        if from_pos and to_pos:
            move_str = f"{from_pos.lower()}{to_pos.lower()}"
            if move_str in rejected_moves:
                print(f"Langkah sebelumnya sudah ditolak: {move_str}")
            else:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        san = board.san(move)
                        board.push(move)
                        log_moves.append(move_str)
                        update_gui(f"{san}")
                        with open("log_langkah.txt", "a") as f:
                            f.write(f"{move_str}\n")
                        print(f"Langkah legal: {san}")
                    else:
                        print(f"Langkah ilegal: {move_str}")
                        rejected_moves.add(move_str)
                except Exception as e:
                    print(f"Error saat push langkah: {e}")
                    rejected_moves.add(move_str)
        else:
            print("Tidak bisa tentukan satu langkah unik.")
        last_state = copy.deepcopy(current_state)
        stable_counter = 0  # reset setelah memproses

    else:
        last_state = copy.deepcopy(current_state)

    # Gabungkan tampilan deteksi dan papan
    board_img = draw_board(board)
    board_img = cv2.resize(board_img, (480, 480))
    combined = np.hstack((annotated_frame, board_img))
    cv2.imshow("Deteksi Catur", combined)

    # Exit dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update GUI Tkinter
    root.update_idletasks()
    root.update()

cap.release()
cv2.destroyAllWindows()
