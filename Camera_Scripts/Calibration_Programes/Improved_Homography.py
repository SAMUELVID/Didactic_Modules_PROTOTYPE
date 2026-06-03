import cv2
import numpy as np
import math
from collections import deque

# --- CAMERA CONFIGURATION ---
camera_Matrix = np.array([[798.73630421,0,316.70047914],[0,795.55356144,248.33415134],[0,0,1]])
distCoeffs = np.array([[0.14503258,-0.45090359,-0.0018612,0.00104603,0.14503459]])
Z_Camera = 600.0

def assign_dynamic_labels(centers, width, height):
    
    corners = [
        (0, 0, "UPPER-LEFT", 0),
        (width, 0, "UPPER-RIGHT", 1),
        (0, height, "DOWN-LEFT", 2),
        (width, height, "DOWN-RIGHT", 3)
    ]

    assigned_points = []
    pts = list(centers)

    while len(pts) > 0:
        
        best_dist = float('inf')
        best_pt = None
        best_corner_idx = -1

        for pt in pts:
            
            for j, c in enumerate(corners):
                
                dist = math.hypot(pt[0] - c[0], pt[1] - c[1])
                
                if dist < best_dist:
                   
                    best_dist = dist
                    best_pt = pt
                    best_corner_idx = j

        assigned_points.append({
            "pt": best_pt,
            "label": corners[best_corner_idx][2],
            "order_idx": corners[best_corner_idx][3]
        })

        pts.remove(best_pt)
        corners.pop(best_corner_idx)

    return assigned_points

def find_centroids(mask, min_area=200, max_area=80000):
    
    h, w = mask.shape
    cv2.rectangle(mask, (0, 0), (w, h), 0, 25)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_objects = []
    
    for c in contours:
        
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)

        if min_area < area < max_area:
            
            M = cv2.moments(hull)
            
            if M["m00"] != 0:
                
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                valid_objects.append({
                    "area": area,
                    "centroid": [cx, cy],
                    "hull": hull
                })

    # Sort ALL detected objects by area (largest to smallest)
    valid_objects = sorted(valid_objects, key=lambda x: x["area"], reverse=True)[:4]

    # Extract only the centroids and hulls of the 4 largest (the actual cubes)
    centroids = [obj["centroid"] for obj in valid_objects]
    valid_contours = [obj["hull"] for obj in valid_objects]

    return centroids, valid_contours

def ask_coordinates(message):
    
    while True:
        
        entry = input(message).replace(',', '.').strip()
        
        if not entry:
            
            continue
        
        try:
            
            value = float(entry)
            
            return value
        
        except ValueError:
            
            print("ERROR. Please enter a valid number.")

# --- INTERACTIVE MENU ---
print("\n" + "="*50)
print(" AUTOCALIBRATION & HOMOGRAPHY ASSISTANT")
print("="*50)
print("1. BLUE Cubes/ Balls Proxy (Height: 27 mm)")
print("2. BLACK Cubes (Height: 49 mm)")
print("==================================================")

while True:
    
    option = input("Choose an option (1 or 2): ")

    if option == '1':
        
        print("\n-> Searching BLUE CUBES...")
        # --- IMPROVEMENT 1: Expanded HSV range for blue ---
        lower_range = np.array([90, 80, 50], np.uint8)
        upper_range = np.array([135, 255, 255], np.uint8)
        color_name = "BLUE CUBES"
        BGR_draw = (255, 0, 0)
        # Contrast color for the text and point (Yellow)
        contrast_color = (0, 255, 255) 
        area_limit = 500
        
        break

    elif option == '2':
        
        print("\n-> Searching BLACK CUBES...")
        lower_range = np.array([0, 0, 0], np.uint8)
        upper_range = np.array([180, 255, 85], np.uint8)
        color_name = "BLACK CUBES"
        BGR_draw = (0, 0, 255)
        # Contrast color for the text and point (Cyan)
        contrast_color = (255, 255, 0) 
        area_limit = 800
        
        break
    
    else:
        
        print("\nNot a valid option. Please select 1 or 2.")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n==================================================")
print(" INSTRUCTIONS:")
print("1. Adjust camera settings if necessary.")
print("2. Wait until the 4 points are stable on the screen.")
print("3. Press 'ENTER' to capture 60 samples and calculate.")
print("4. Press 'ESC' to exit without saving.")
print("==================================================\n")

NUM_CALIB_SAMPLES = 60
captured_history = {i: {"x": [], "y": []} for i in range(4)}
capturing = False
final_ordered_points = None

while True:

    ret, frame = cap.read()

    if not ret:

        print("Error reading camera.")
        
        break

    frame = cv2.undistort(frame, camera_Matrix, distCoeffs)
    
    h, w = frame.shape[:2]

    # --- IMPROVEMENT 2: Green central crosshair covering the entire axis ---
    cx_cross, cy_cross = int(w / 2), int(h / 2)
    cv2.line(frame, (cx_cross, 0), (cx_cross, h), (0, 255, 0), 1)
    cv2.line(frame, (0, cy_cross), (w, cy_cross), (0, 255, 0), 1)
    # -------------------------------------------------------------

    softened_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(softened_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    if option == '1':
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel_balls = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel_balls, iterations=2)
        mask = cv2.erode(mask, kernel_balls, iterations=2)
   
    else:
        
        kernel_open = np.ones((5, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((25, 25), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    centers, hulls = find_centroids(mask, min_area=area_limit)

    if len(centers) == 4:
        
        assigned = assign_dynamic_labels(centers, w, h)

        if capturing:
            
            for item in assigned:
                
                idx = item["order_idx"]
                raw_pt = item["pt"]
                captured_history[idx]["x"].append(raw_pt[0])
                captured_history[idx]["y"].append(raw_pt[1])
            
            progress = len(captured_history[0]["x"])
            cv2.putText(frame, f"CALIBRATING: {progress}/{NUM_CALIB_SAMPLES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            for item in assigned:
               
                pt = item["pt"]
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            if progress >= NUM_CALIB_SAMPLES:
                
                final_ordered_points = []
                
                for i in range(4):
                    
                    avg_x = sum(captured_history[i]["x"]) / NUM_CALIB_SAMPLES
                    avg_y = sum(captured_history[i]["y"]) / NUM_CALIB_SAMPLES
                    final_ordered_points.append([avg_x, avg_y])
               
                print("\n60 frames captured and averaged successfully!")
               
                break

        else:
            
            # Standard cube drawing
            for item in assigned:
                
                raw_pt = item["pt"]
                label = item["label"]
                pt_x = int(raw_pt[0])
                pt_y = int(raw_pt[1])

                # --- IMPROVEMENT 3: Points and coordinate labels ---
                # Contrasted inner circle (Yellow/Cyan)
                cv2.circle(frame, (pt_x, pt_y), 4, contrast_color, -1)
                # Outer ring with cube color (optional, for highlighting)
                cv2.circle(frame, (pt_x, pt_y), 7, BGR_draw, 1)

                # Original label ("UPPER-LEFT")
                cv2.putText(frame, label, (pt_x - 40, pt_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR_draw, 2)
                
                # Exact coordinate label
                coord_text = f"X:{pt_x} Y:{pt_y}"
                cv2.putText(frame, coord_text, (pt_x + 10, pt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, contrast_color, 2)
                # --------------------------------------------------

            cv2.putText(frame, "4 DETECTED! Press 'Enter' to confirm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        
        if capturing:
           
            print("\nLost track of the 4 cubes during capture! Aborting...")
            capturing = False
            
            for i in range(4):
                
                captured_history[i]["x"].clear()
                captured_history[i]["y"].clear()

    # --- ON-SCREEN UI OVERLAY ---
    cv2.circle(frame, (0, 0), 15, (0, 255, 255), -1)
    
    cv2.line(frame, (0, 0), (100, 0), (0, 0, 255), 5)
    cv2.line(frame, (0, 0), (0, 100), (0, 255, 0), 5)
    
    cv2.putText(frame, "Origin (0,0)", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, "RED Line: X Axis (+X to the right)", (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.putText(frame, "GREEN Line: Y Axis (+Y downwards)", (w - 250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(frame, "UL: Nearest to (0,0)", (w - 250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(frame, "UR: Nearest to (+X,0)", (w - 250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(frame, "DL: Nearest to (0,+Y)", (w - 250, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(frame, "DR: Furthest Away (+X,+Y)", (w - 250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(frame, "ESC Key to exit", (w - 250, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.namedWindow('Homography Assistant',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
    cv2.imshow('Homography Assistant', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF

    if key == 13 and not capturing:
        
        if len(centers) == 4:
           
            print("\nCapturing 60 stable samples. Please wait...")
            capturing = True
        
        else:
          
            print("\nERROR: Make sure 4 cubes are detected before capturing.")
    
    elif key == 27:
        
        print("\nCancelled by user.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# --- HOMOGRAPHY CALCULATION ---
if final_ordered_points is not None:
   
    print("\n")
    names = ["UPPER-LEFT", "UPPER-RIGHT", "DOWN-LEFT", "DOWN-RIGHT"]

    for i, pt in enumerate(final_ordered_points):
        
        print(f"[{names[i]}] Pixel saved: X = {pt[0]:.2f}, Y = {pt[1]:.2f}")

    print("\nPixels registered correctly!")
    print("Enter the REAL coordinates (in mm) read from the robot TCP or measured by hand.")
    print("-> Remember to subtract the scanning position to obtain the camera relative position\n")

    real_points = []
    
    for i in range(4):
       
        print(f"--- Point {i+1}: {names[i]} ---")
        x_mm = ask_coordinates("Enter real X (mm): ")
        y_mm = ask_coordinates("Enter real Y (mm): ")
        real_points.append([x_mm, y_mm])

    if option == '1':
        
        Z_Obj = 27.0
    else:
        
        Z_Obj = 49.0

    scale = (Z_Camera - Z_Obj) / Z_Camera
    Cx = camera_Matrix[0, 2]
    Cy = camera_Matrix[1, 2]

    corrected_points = []

    for pt in final_ordered_points:
       
        x_px = pt[0]
        y_px = pt[1]

        x_corr = Cx + (x_px - Cx) * scale
        y_corr = Cy + (y_px - Cy) * scale
        corrected_points.append([x_corr, y_corr])

    points_src = np.array(corrected_points, dtype=np.float32)
    points_dst = np.array(real_points, dtype=np.float32)

    H, status = cv2.findHomography(points_src, points_dst)

    if H is not None:
       
        print("\n" + "="*50)
        print(f" Calculated Matrix for: {color_name}")
        print("="*50)
        print(f"H11 = {H[0,0]:.8f}")
        print(f"H12 = {H[0,1]:.8f}")
        print(f"H13 = {H[0,2]:.8f}")
        print("-" * 18)
        print(f"H21 = {H[1,0]:.8f}")
        print(f"H22 = {H[1,1]:.8f}")
        print(f"H23 = {H[1,2]:.8f}")
        print("-" * 18)
        print(f"H31 = {H[2,0]:.8f}")
        print(f"H32 = {H[2,1]:.8f}")
        print(f"H33 = {H[2,2]:.8f}")
        print("="*50)
    
    else:
        
        print("\n[!] CRITICAL ERROR. Could not calculate matrix.")
        print("Make sure the numbers introduced are part of a real area.")

else:
    
    print("\nCalibration canceled or incomplete")