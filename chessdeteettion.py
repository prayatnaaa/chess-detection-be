import cv2
import numpy as np
from ultralytics import YOLO
import chess
import chess.svg
import cairosvg
import tkinter as tk
import copy

# --- Initialization ---
board = chess.Board()
model = YOLO('chess_model/best.pt') 
log_moves = []
rejected_moves = set()

# ## <<< CHANGE: Renamed last_state to be more descriptive and added stable_last_state
last_frame_state = {}         # State from the immediately preceding frame
stable_last_state = {}        # The last known stable board configuration
stable_counter = 0
required_stable_frames = 5    # Increased for better stability

# --- Tkinter GUI Setup ---
root = tk.Tk()
# root.title("Chess Move Log")
listbox = tk.Listbox(root, width=30, height=30, font=("Courier", 12))
listbox.pack(padx=10, pady=10)

def update_gui(move_str):
    """Updates the Tkinter listbox with a new move."""
    print(f"Update GUI: {move_str}")
    listbox.insert(tk.END, move_str)
    listbox.see(tk.END)

update_gui("GUI Ready")

# --- Image and Board Functions ---
def draw_board(board, size=480):
    """Generates a CV2 image from the current python-chess board state."""
    board_svg = chess.svg.board(board=board, size=size)
    png_bytes = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), output_width=size, output_height=size)
    png_arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(png_arr, cv2.IMREAD_COLOR)
    return img

def to_notation(row, col):
    """Converts grid coordinates (0-7) to chess notation (e.g., h1)."""
    # Note: This mapping assumes (row=0, col=0) is the h1 square.
    # Standard is (0,0) -> a8. You may need to adjust this.
    # Standard mapping: file = chr(ord('a') + col), rank = str(8 - row)
    file = chr(ord('a') + (7 - col)) 
    rank = str(1 + row)
    return file + rank

def get_piece_positions(results):
    """Extracts piece positions from YOLO results."""
    piece_pos = {}
    if not results or not results[0].boxes:
        return piece_pos

    boxes = results[0].boxes
    cell_size = results[0].orig_img.shape[0] // 8

    for i in range(len(boxes)):
        if boxes.conf[i] < 0.4: # Increased confidence threshold
            continue
        
        xyxy = boxes.xyxy[i].cpu().numpy()
        x_center = int((xyxy[0] + xyxy[2]) / 2)
        y_center = int((xyxy[1] + xyxy[3]) / 2)
        
        col = min(x_center // cell_size, 7)
        row = min(y_center // cell_size, 7)
        
        notation = to_notation(row, col)
        piece_pos[notation] = True
    
    return piece_pos

def detect_move(prev, curr):
    """Compares two board states to find a single move."""
    removed = [pos for pos in prev if pos not in curr]
    added = [pos for pos in curr if pos not in prev]

    print(f"State Change Analysis - Removed: {removed}, Added: {added}")

    # This handles standard moves. It does not handle castling or en passant.
    if len(removed) == 1 and len(added) == 1:
        return removed[0], added[0]
    return None, None

# --- Main Video Loop ---
cap = cv2.VideoCapture('chessvid.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or video error.")
        break

    frame_resized = cv2.resize(frame, (480, 480))
    results = model(frame_resized)
    current_state = get_piece_positions(results)

    # Annotate the video frame with detection boxes for debugging
    annotated_frame = frame_resized.copy()
    if results[0].boxes:
        for box in results[0].boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ## <<< CHANGE: The entire logic for detecting moves is new
    # Check for frame-to-frame stability
    if current_state == last_frame_state:
        stable_counter += 1
    else:
        stable_counter = 0 # Reset if the scene is changing

    # If the scene has been stable for enough frames AND it's a NEW stable state
    if stable_counter >= required_stable_frames and current_state != stable_last_state:
        print("New stable state detected. Analyzing move...")
        
        from_pos, to_pos = detect_move(stable_last_state, current_state)

        if from_pos and to_pos:
            move_str = f"{from_pos.lower()}{to_pos.lower()}"
            if move_str in rejected_moves:
                print(f"Move already rejected: {move_str}")
            else:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        san = board.san(move)
                        board.push(move) 
                        log_moves.append(move_str)
                        update_gui(f"{len(log_moves)}. {san}")
                        
                        with open("log_langkah.txt", "a") as f:
                            f.write(f"{move_str}\n")
                        print(f"LEGAL MOVE: {san}")
                    else:
                        print(f"Illegal move detected: {move_str}")
                        rejected_moves.add(move_str)
                except Exception as e:
                    print(f"Error processing move '{move_str}': {e}")
                    rejected_moves.add(move_str)

        # CRITICAL: Update the stable state to the new current state
        stable_last_state = copy.deepcopy(current_state)

    # Update the last frame's state for the next iteration's stability check
    last_frame_state = copy.deepcopy(current_state)
    
    # --- Display ---
    board_img = draw_board(board) # This now uses the potentially updated board
    combined_view = np.hstack((annotated_frame, board_img))
    cv2.imshow("Chess Detection", combined_view)

    # --- GUI and Exit Condition ---
    root.update_idletasks()
    root.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()