import cv2
import numpy as np
import time
import threading
from pyModbusTCP.server import ModbusServer
from flask import Flask, Response

# --- 1. CAMERA CALIBRATION PARAMETERS ---
# Intrinsic matrix and distortion coefficients to correct lens deformation
camera_Matrix = np.array([[798.73630421,0,316.70047914],[0,795.55356144,248.33415134],[0,0,1]])
distCoeffs = np.array([[0.14503258,-0.45090359,-0.0018612,0.00104603,0.14503459]])

# --- 2. FLASK WEB SERVER CONFIGURATION ---
app = Flask(__name__)
static_frame = None

def decode_modbus_dint(high_word, low_word):

    # Reconstructs a 32-bit integer from two 16-bit Modbus registers
    raw_32 = (high_word << 16) | low_word

    # Handle two's complement for negative values
    if raw_32 > 2147483647:

        raw_32 -= 4294967296

    return raw_32 

def slow_photo_generator():

    # Yields static frames at a reduced framerate for the web interface
    global static_frame

    while True:

        if static_frame is not None:

            ret, jpeg = cv2.imencode('.jpg', static_frame)

            if ret:

                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            else:

                # Fallback empty image if encoding fails
                emp_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', emp_img)
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            # Throttle refresh rate to save bandwidth
            time.sleep(0.5)

@app.route('/hmi')
def hmi():

    # Renders the main HTML page for the HMI web view
    html = """
        <html>
            <body style="margin:0; background-color:black; display:flex; justify-content:center; align-items:center;">
                <img src ="/stream" style="width:100%; height:100%; object-fit:fill;">
            </body>
        </html>
    """

    return html

@app.route('/stream')
def stream():

    # Route providing the MJPEG stream
    return Response(slow_photo_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():

    # Launches the Flask server on a background thread
    print("Trying to run web server at http://127.0.0.1:5005/hmi...")
    app.run(host='0.0.0.0', port=5005, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# --- 3. MODBUS TCP SERVER INITIALIZATION ---
server = ModbusServer("127.0.0.1", 502, no_block=True)
server.start()
print("Modbus TCP initialized at 127.0.0.1:502")

# --- 4. CAMERA HARDWARE SETUP ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Resolution downscaling for performance optimization
# En caso de colapso poner a 640 (Original 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# En caso de colapso poner a 480 (Original 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 5. COLOR THRESHOLDS & MODBUS REGISTRY MAPPING ---
colors = {
    
    "RED": {
        "Reg_Start": 0,
        "Reg_Read": 200,
        "BGR": (0, 0, 255),
        "Ranges": {
            "R1": {
                "upper": np.array([4, 255, 255], np.uint8),
                "lower": np.array([0, 100, 100], np.uint8)
            },
            "R2": {
                "upper": np.array([179, 255, 255], np.uint8),
                "lower": np.array([172, 100, 100], np.uint8)
            }
        }
    },

    "GREEN": {
        "Reg_Start": 19,
        "Reg_Read": 212,
        "BGR": (0, 255, 0),
        "Ranges": {
            "R1": {
                "upper": np.array([85, 255, 255], np.uint8),
                "lower": np.array([40, 80, 60], np.uint8)
            }
        }
    },

    "BLUE": {
        "Reg_Start": 38,
        "Reg_Read": 224,
        "BGR": (255, 0, 0),
        "Ranges": {
            "R1": {
                "upper": np.array([135, 255, 255], np.uint8),
                "lower": np.array([100, 100, 100], np.uint8)
            }
        }
    },

    "BLACK": {
        "Reg_Start": 57,
        "BGR": (0, 0, 0),
        "Ranges": {
            "R1": {
                "upper": np.array([180, 255, 85], np.uint8),
                "lower": np.array([0, 0, 0], np.uint8)
            }
        }
    },

    "YELLOW": {
        "Reg_Start": 66,
        "Reg_Read": 236,
        "BGR": (0, 255, 255),
        "Ranges": {
            "R1": {
                "upper": np.array([35, 255, 255], np.uint8),
                "lower": np.array([17, 80, 80], np.uint8)
            }
        }
    },

    "ORANGE": {
        "Reg_Start": 85,
        "Reg_Read": 248,
        "BGR": (0, 165, 255),
        "Ranges": {
            "R1": {
                "upper": np.array([11, 255, 255], np.uint8),
                "lower": np.array([6, 120, 120], np.uint8)
            }
        }
    }
}

# --- 6. STATE VARIABLES ---
prev_gray = None
static_counter = 0
MOVEMENT_STATE = 1
STATE_REGISTRY = 104
box_smoothing = {}

cv2.namedWindow('VOXICON Panel AI - Modbus Server', cv2.WINDOW_NORMAL)

# --- 7. MAIN VISION LOOP ---
while True:

    ret, frame = cap.read()

    if not ret:

        print("Critical Error: No response from camera")
        break
    
    # Image distortion correction and noise filtering
    frame = cv2.undistort(frame, camera_Matrix, distCoeffs)
    softened_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(softened_frame, cv2.COLOR_BGR2GRAY)

    # Initialize previous frame for motion detection
    if prev_gray is None:

        prev_gray = gray

    # Motion detection by evaluating absolute difference between consecutive frames
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh)
    prev_gray = gray

    # State machine for movement detection (Dynamic vs Static scene)
    if changed_pixels > 3000:

        static_counter = 0
        MOVEMENT_STATE = 1

    else:

        static_counter += 1
        
        if static_counter > 15:

            MOVEMENT_STATE = 0

    # Sync movement state to Modbus registry
    server.data_bank.set_holding_registers(STATE_REGISTRY, [MOVEMENT_STATE])
    
    # Convert to HSV color space for robust color isolation
    hsv = cv2.cvtColor(softened_frame, cv2.COLOR_BGR2HSV)
    clean_frame_HMI = frame.copy()
    new_box_smoothing = {}

    text = 35

    # --- 8. COLOR PROCESSING PIPELINE ---
    for color_name, data in colors.items():

        new_box_smoothing[color_name] = []
        end_mask = np.zeros(frame.shape[:2], dtype="uint8")

        # Combine multiple HSV ranges for colors wrapping around the spectrum (e.g., Red)
        for range_name, limit in data["Ranges"].items():

            upper = limit["upper"]
            lower = limit["lower"]
            temp_mask = cv2.inRange(hsv, lower, upper)
            end_mask = cv2.bitwise_or(end_mask, temp_mask)
        
        # Custom morphological operations for specific colors
        if color_name == "BLACK":

            # Aggressive filtering for dark objects
            kernel_open = np.ones((5, 15), np.uint8)
            end_mask = cv2.morphologyEx(end_mask, cv2.MORPH_OPEN, kernel_open)
            kernel_close = np.ones((25, 25), np.uint8)
            end_mask = cv2.morphologyEx(end_mask, cv2.MORPH_CLOSE, kernel_close)
            cv2.namedWindow("Debugging Window",cv2.WINDOW_NORMAL)
            cv2.imshow("Debugging Window",end_mask)

        else:

            # Standard filtering for bright colors
            kernel = np.ones((5, 5), np.uint8)
            end_mask = cv2.morphologyEx(end_mask, cv2.MORPH_CLOSE, kernel)
            kernel_balls = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            end_mask = cv2.dilate(end_mask, kernel_balls, iterations=2)
            end_mask = cv2.erode(end_mask, kernel_balls, iterations=2)

        # Find object contours in the generated mask
        contours, _ = cv2.findContours(end_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        
        # --- 9. GEOMETRIC FILTERING ---
        for c in contours:

            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)

            if color_name == "BLACK":

                if 800 < area < 80000:

                    x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(hull)
                    # Añadido 1e-5 para evitar error de división por cero
                    aspect_ratio = float(w_tmp) / (h_tmp + 1e-5)

                    # Filter by aspect ratio to discard elongated artifacts
                    if 0.4 < aspect_ratio < 2.5: 

                        valid_contours.append(hull)

            else:

                if area > 2000:

                    valid_contours.append(hull)

        # Sort identified objects by size (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        detected_pieces = 0
        geometric_data = []
        reg_base = data["Reg_Start"]
        reg_read = data.get("Reg_Read")

        # --- 10. OBJECT TRACKING & DATA EXTRACTION ---
        for hull_contour in valid_contours:

            M = cv2.moments(hull_contour)

            if M["m00"] != 0:

                detected_pieces += 1
                
                # Calculate raw centroid coordinates
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                x, y, w, h = cv2.boundingRect(hull_contour)

                best_match = None
                min_dist = float('inf')

                # Euclidean distance matching to identify previously tracked objects
                if color_name in box_smoothing:

                    for old_box in box_smoothing[color_name]:

                        old_x, old_y, old_w, old_h = old_box[:4]

                        old_cx = old_x + (old_w / 2)
                        old_cy = old_y + (old_h / 2)

                        dist = ((cx - old_cx)**2 + (cy - old_cy)**2)**0.5

                        if dist < 125 and dist < min_dist:

                            min_dist = dist
                            best_match = old_box

                # Dynamic data smoothing (Alpha filter)
                if best_match is not None:

                    if len(best_match) >= 8:

                        old_x, old_y, old_w, old_h, last_vx, last_vy, last_cx, last_cy = best_match[:8]

                    elif len(best_match) == 6:

                        old_x, old_y, old_w, old_h, last_vx, last_vy = best_match
                        last_cx, last_cy = cx, cy

                    else:

                        old_x, old_y, old_w, old_h = best_match[:4]
                        last_vx, last_vy = 0, 0
                        last_cx = cx
                        last_cy = cy

                    current_vx = x - old_x
                    current_vy = y - old_y
                    speed = (current_vx**2 + current_vy**2)**0.5

                    # --- LÓGICA DE FILTRADO DINÁMICO ---
                    if speed < 1.0:

                        ALPHA = 0.01  # Congelación máxima en estático

                    elif speed > 5.0:

                        ALPHA = 0.85  # Respuesta rápida en movimiento

                    else:

                        ALPHA = 0.4   # Transición normal

                    # Apply filter coefficients to smooth tracking box and centroid
                    x_smooth = int((ALPHA * x) + ((1.0 - ALPHA) * old_x))
                    y_smooth = int((ALPHA * y) + ((1.0 - ALPHA) * old_y))
                    w_smooth = int((ALPHA * w) + ((1.0 - ALPHA) * old_w))
                    h_smooth = int((ALPHA * h) + ((1.0 - ALPHA) * old_h))
                    cx_smooth = (ALPHA * cx) + ((1.0 - ALPHA) * last_cx)
                    cy_smooth = (ALPHA * cy) + ((1.0 - ALPHA) * last_cy)

                    new_vx = x_smooth - old_x
                    new_vy = y_smooth - old_y
                    new_box_smoothing[color_name].append([x_smooth, y_smooth, w_smooth, h_smooth, new_vx, new_vy, cx_smooth, cy_smooth])
                    
                    x, y, w, h = x_smooth, y_smooth, w_smooth, h_smooth
                    cx, cy = cx_smooth, cy_smooth

                else:

                    # Register new object if no tracking match is found
                    new_box_smoothing[color_name].append([x, y, w, h, 0, 0, cx, cy])

                # Transformación a RAW Pixels multiplicados por 100 para Modbus
                cx_modbus = int(cx * 100)
                cy_modbus = int(cy * 100)

                # Annotate frames with bounding boxes
                if color_name != "BLACK":

                    cv2.rectangle(frame, (x, y), (x + w, y + h), data["BGR"], 1)
                    cv2.rectangle(clean_frame_HMI, (x, y), (x + w, y + h), data["BGR"], 1)

                # Build Modbus payload array
                if color_name == "BLACK":

                    geometric_data.extend([cx_modbus, cy_modbus])

                else:

                    geometric_data.extend([x, y, w, h, cx_modbus, cy_modbus])

        # Update count of detected pieces per color in Modbus registry
        server.data_bank.set_holding_registers(reg_base, [detected_pieces])

        # Push geometric arrays to Modbus banks, zero-padding missing data
        if len(geometric_data) > 0:

            if color_name == "BLACK":

                complete_data = geometric_data[:8]

                while len(complete_data) < 8:

                    complete_data.append(0)

            else:

                complete_data = geometric_data[:18]

                while len(complete_data) < 18:

                    complete_data.append(0)

            server.data_bank.set_holding_registers(reg_base + 1, complete_data)

        else:

            zeros_limit = 8 if color_name == "BLACK" else 18
            server.data_bank.set_holding_registers(reg_base + 1, [0] * zeros_limit)

        # --- 11. HMI VISUALIZATION OVERLAYS ---
        if detected_pieces > 0 and color_name != "BLACK":

            # Print summary statistics on screen
            cv2.putText(frame, f"{color_name} BALLS", (520, text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, data["BGR"], 1)
            cv2.putText(clean_frame_HMI, f"{color_name} BALLS", (520, text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, data["BGR"], 1)
            text += 25

            # Read feedback coordinates from PLC via Modbus for visual verification
            if reg_read is not None:

                reg_mm = server.data_bank.get_holding_registers(reg_read, 12)

                if reg_mm:

                    for i in range(min(detected_pieces, 3)):

                        base_id = i * 4
                        cx_raw = decode_modbus_dint(reg_mm[base_id], reg_mm[base_id + 1])
                        cy_raw = decode_modbus_dint(reg_mm[base_id + 2], reg_mm[base_id + 3])
                        cx_mm = cx_raw / 1000.0
                        cy_mm = cy_raw / 1000.0

                        cv2.putText(frame, f"B{i+1}: X = {cx_mm:.2f}", (540, text), cv2.FONT_HERSHEY_SIMPLEX, 0.3, data["BGR"], 1)
                        cv2.putText(clean_frame_HMI, f"B{i+1}: X = {cx_mm:.2f}", (540, text), cv2.FONT_HERSHEY_SIMPLEX, 0.3, data["BGR"], 1)
                        text += 20

                        cv2.putText(frame, f"    Y = {cy_mm:.2f}", (540, text), cv2.FONT_HERSHEY_SIMPLEX, 0.3, data["BGR"], 1)
                        cv2.putText(clean_frame_HMI, f"    Y = {cy_mm:.2f}", (540, text), cv2.FONT_HERSHEY_SIMPLEX, 0.3, data["BGR"], 1)
                        text += 25

                text += 10

    # Snapshot saving mechanism when movement ceases
    if MOVEMENT_STATE == 0:

        static_frame = clean_frame_HMI.copy()

    # Display system status UI
    text_state = "WAITING... (MOVING)" if MOVEMENT_STATE == 1 else "STATIC PHOTO (STABLE)"
    color_text = (0, 0, 255) if MOVEMENT_STATE == 1 else (0, 255, 0)
    
    cv2.putText(frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 2)
    
    # Propagate tracking arrays to the next loop iteration
    box_smoothing = new_box_smoothing

    cv2.imshow('VOXICON Panel AI - Modbus Server', frame)

    # Main kill switch (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# --- 12. CAMERA SHUTDOWN ---
cap.release()
server.stop()
cv2.destroyAllWindows()