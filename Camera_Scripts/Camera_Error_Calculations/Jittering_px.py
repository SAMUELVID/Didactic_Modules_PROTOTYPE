import cv2
import numpy as np

# --- 1. CAMERA CONFIGURATION ---
camera_Matrix = np.array([[798.73630421,0,316.70047914],[0,795.55356144,248.33415134],[0,0,1]])
distCoeffs = np.array([[0.14503258,-0.45090359,-0.0018612,0.00104603,0.14503459]])

# --- 2. INTERACTIVE MENU ---
print("==================================================")
print(" JITTERING TEST SCRIPT (CAMERA NOISE)")
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

NUM_SAMPLES = 100
samples_x = []
samples_y = []

print(f"\nCollecting {NUM_SAMPLES} frames to compute the Jitter for the {color_name} cube...")
print("DO NOT MOVE THE CAMERA OR THE CUBE.")
print("Press 'ESC' to cancel the test.\n")

# --- 3. CAPTURE LOOP ---
while len(samples_x) < NUM_SAMPLES:

    ret, frame = cap.read()

    if not ret:

        continue

    # Undistort (Vital to keep pixels accurate)
    frame = cv2.undistort(frame, camera_Matrix, distCoeffs)
    
    # Initial processing
    softened = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(softened, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # --- 4. ADAPTIVE MORPHOLOGICAL FILTERS ---
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

    # --- 5. CENTROID EXTRACTION ---
    if contours:

        # Sort from largest to smallest area to avoid shadows
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:

            area = cv2.contourArea(c)
            is_valid = False

            # Area validation criteria by color
            if option == '1' and area > 500:

                is_valid = True

            elif option == '2':

                if 800 < area < 80000:

                    x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(c)
                    aspect_ratio = float(w_tmp) / (h_tmp + 1e-5)
                    
                    # Expanded Aspect Ratio for the Jittering test
                    if 0.2 < aspect_ratio < 4.0:

                        is_valid = True

            if is_valid:

                M = cv2.moments(c)

                if M["m00"] != 0:

                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    samples_x.append(cx)
                    samples_y.append(cy)

                    # Visual feedback
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    cv2.putText(frame, f"Samples: {len(samples_x)}/{NUM_SAMPLES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Exit the loop when the correct piece is found
                    break

    cv2.imshow(f'Test Jitter - {color_name}', frame)
    cv2.imshow('Mask', mask)

    # Use ESC key (27 in ASCII) to exit
    if cv2.waitKey(1) & 0xFF == 27:

        break

cap.release()
cv2.destroyAllWindows()

# --- 6. STATISTICAL CALCULATION ---
if len(samples_x) == NUM_SAMPLES:

    sigma_x = np.std(samples_x)
    sigma_y = np.std(samples_y)
    
    print("\n==================================================")
    print(" TEST COMPLETED SUCCESSFULLY")
    print("==================================================")
    print(f" Tested Object     : {color_name} Cube")
    print(f" Deviation (sigma) X  : {sigma_x:.4f} px")
    print(f" Deviation (sigma) Y  : {sigma_y:.4f} px")
    print("==================================================")
    print(" Write these values in your Excel and multiply them")
    print(" by your scale factor (mm/px) to get e_cam.")

else:

    print("\n Test cancelled by the user.")