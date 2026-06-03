import cv2
import numpy as np

# --- 1. CAMERA CONFIGURATION ---
camera_Matrix = np.array([[798.73630421,0,316.70047914],[0,795.55356144,248.33415134],[0,0,1]])
distCoeffs = np.array([[0.14503258,-0.45090359,-0.0018612,0.00104603,0.14503459]])

# --- 2. INTERACTIVE MENU ---
print("==================================================")
print(" RAW PIXEL EXTRACTION FOR EXCEL")
print("==================================================")
print("Select the object to test:")
print("1. BLUE Cube / Balls Proxy (Height: 27 mm)")
print("2. BLACK Cube (Height: 49 mm)")
print("==================================================")

option = input("Enter 1 or 2: ")

if option == '1':

    color_name = "BLUE"
    lower_color = np.array([100, 100, 100], np.uint8)
    upper_color = np.array([135, 255, 255], np.uint8)

elif option == '2':

    color_name = "BLACK"
    lower_color = np.array([0, 0, 0], np.uint8)
    upper_color = np.array([180, 255, 85], np.uint8)

else:

    print("Invalid option. Closing program.")
    exit()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

point1 = None
point2 = None

print("\n==================================================")
print(" INSTRUCTIONS:")
print("1. Put the cube in position and press 'SPACE' for Pos 1.")
print("2. Move the cube and press 'SPACE' for Pos 2.")
print("3. Press 'R' to reset and measure another pair.")
print("4. Press 'Q' or 'ESC' to exit.")
print("==================================================\n")

while True:

    ret, frame = cap.read()

    if not ret:

        print("Error reading camera.")
        break

    # 1. UNDISTORT (Maintains accurate pixels)
    frame = cv2.undistort(frame, camera_Matrix, distCoeffs)
    
    # 2. IMAGE PROCESSING
    softened_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(softened_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # --- 3. ADAPTIVE MORPHOLOGICAL FILTERS ---
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_detected = False
    raw_cx, raw_cy = 0.0, 0.0

    if contours:

        # Sort from largest to smallest area to avoid shadows
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours:

            area = cv2.contourArea(c)
            is_valid = False
        
            if option == '1' and area > 500:

                is_valid = True

            elif option == '2':

                if 800 < area < 80000:

                    x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(c)
                    aspect_ratio = float(w_tmp) / (h_tmp + 1e-5)

                    if 0.4 < aspect_ratio < 2.5:

                        is_valid = True

            # If it passes the filters, we draw and break the loop
            if is_valid:

                M = cv2.moments(c)

                if M["m00"] != 0:

                    object_detected = True
                    raw_cx = M["m10"] / M["m00"]
                    raw_cy = M["m01"] / M["m00"]

                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    c_x_int, c_y_int = int(raw_cx), int(raw_cy)
                    cv2.line(frame, (c_x_int - 10, c_y_int), (c_x_int + 10, c_y_int), (0, 0, 255), 2)
                    cv2.line(frame, (c_x_int, c_y_int - 10), (c_x_int, c_y_int + 10), (0, 0, 255), 2)

                    cv2.putText(frame, f"Current -> X: {raw_cx:.3f} | Y: {raw_cy:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Exit the loop when the correct piece is found
                    break 

    # Screen UI points
    if point1 is not None:

        cv2.putText(frame, f"Pos 1: X={point1[0]:.3f}, Y={point1[1]:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if point2 is not None:

        cv2.putText(frame, f"Pos 2: X={point2[0]:.3f}, Y={point2[1]:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(f'Raw Pixels - {color_name}', frame)
    cv2.imshow('Mask', mask)

    # 4. CONTROLS
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27: # Q or ESC

        break

    elif key == ord('r'):

        point1 = None
        point2 = None
        print("\nReset. Ready for new measurements.")

    elif key == 32: # SPACE

        if object_detected:

            if point1 is None:

                point1 = (raw_cx, raw_cy)
                print(f"Position 1 -> X: {raw_cx:.3f} , Y: {raw_cy:.3f}")

            elif point2 is None:

                point2 = (raw_cx, raw_cy)
                print(f"Position 2 -> X: {raw_cx:.3f} , Y: {raw_cy:.3f}")

        else:

            print("Object not detected.")

cap.release()
cv2.destroyAllWindows()