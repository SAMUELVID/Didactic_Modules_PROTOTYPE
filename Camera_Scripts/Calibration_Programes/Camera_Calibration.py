import cv2
import numpy as np
import math
import time

print("Initialising Camera Calibration Asistant...")

# --- 1. CAMERA CONFIGURATION ---
W = 640
H = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
time.sleep(1)

if not cap.isOpened():

	print ("[FATAL ERROR] Can not connect to the camera")
	exit()

# --- 2. CHESSBOARD PATTERN CONFIGURATION ---
PATRON = (7,7)

# --- 3. MAIN ALIGNMENT LOOP ---
while True:

	ret, frame = cap.read()
	if not ret: break

	# Convert frame to grayscale for corner detection
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	height,wide = frame.shape[:2]
	scale = wide/640.0
	font_size = 0.4 * scale
	thickness = max(1, int(2*scale))
	cross_ratio = int(20 * scale)
	pos_x = int(10* scale)
	line_1 = int(30 * scale)
	line_2 = int(70 * scale)
	line_3 = int(110 * scale)
	px_tolerance = int(10 * scale)
	cx_screen = wide // 2
	cy_screen = height // 2
	
	# Draw central mathematical crosshair
	cv2.line(frame, (cx_screen - cross_ratio, cy_screen), (cx_screen + cross_ratio, cy_screen), (0,0,255), thickness)
	cv2.line(frame, (cx_screen, cy_screen - cross_ratio), (cx_screen, cy_screen + cross_ratio), (0,0,255), thickness)
	
	# Attempt to find chessboard corners
	ret_board, corners = cv2.findChessboardCorners(gray, PATRON, None)

	if ret_board:

		# Sub-pixel refinement for maximum accuracy
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
		cv2.drawChessboardCorners(frame,PATRON, corners, ret_board)
		
		# Extract top-left and top-right corners to calculate rotation angle
		tl = corners[0][0]
		tr = corners[PATRON[0] - 1][0]
		dy = tr[1] - tl[1]
		dx = tr[0] - tl[0]
		rad_angle = math.atan2(dy, dx)
		deg_angle = math.degrees(rad_angle)

		# Normalize the Z-axis rotation angle (Roll).
        # Since a chessboard has 90-degree rotational symmetry, we restrict 
        # the alignment angle to a [-45, 45] degree range. This ensures the 
        # system always prompts the shortest rotation path to straighten the camera.
        # The condition is repeated as a simple fallback to handle larger initial 
        # angles without needing a while loop.
		if deg_angle > 45: deg_angle -= 90
		elif deg_angle < -45: deg_angle += 90 
		if deg_angle > 45: deg_angle -= 90
		elif deg_angle < -45: deg_angle += 90

		# Evaluate Z-axis rotation (Roll)
		if deg_angle > 0.5:

			txt_z = f"AXIS Z: TURN CAMERA CLOCKWISE ({deg_angle:.2f} deg)"
			color_z = (0, 0, 255)

		elif deg_angle < -0.5:

			txt_z = f"AXIS Z: TURN CAMERA COUNTER-CLOCKWISE ({deg_angle:.2f}) deg"
			color_z = (0, 0, 255)

		else:

			txt_z = f"AXIS Z: PERFECT({deg_angle:.2f} deg)"
			color_z = (0,255,0)

		# Calculate the geometric center of the chessboard
		cx_board = int(np.mean(corners[:,0,0]))
		cy_board = int(np.mean(corners[:,0,1]))

		cv2.circle(frame, (cx_board, cy_board), int(8* scale), (0,255,0), -1)
		
		# Calculate pixel offset from screen center
		offset_x = cx_screen - cx_board
		offset_y = cy_screen - cy_board

		# Evaluate X-axis offset
		if offset_x > px_tolerance:

			txt_x = f"AXIS X: MOVE THE CAMERA TO THE RIGTH({offset_x} px)"
			color_x = (0, 165, 255)

		elif offset_x < -px_tolerance:

			txt_x = f"AXIS X: MOVE THE CAMERA TO THE LEFT({offset_x} px)"
			color_x = (0, 165, 255)

		else:

			txt_x = "AXIS X: CENTERED"
			color_x = (0,255,0)

		# Evaluate Y-axis offset
		if offset_y > px_tolerance:

			txt_y = f"AXIS Y: MOVE THE CAMERA DOWN({offset_y} px)"
			color_y = (0, 165, 255)

		elif offset_y < -px_tolerance:

			txt_y = f"AXIS Y: MOVE THE CAMERA UP({offset_y} px)"
			color_y = (0, 165, 255)

		else:

			txt_y = "AXIS Y: CENTERED"
			color_y = (0,255,0)

		# Display alignment instructions on screen
		cv2.putText(frame, txt_z, (pos_x,line_1),cv2.FONT_HERSHEY_SIMPLEX, font_size, color_z, thickness)
		cv2.putText(frame, txt_x, (pos_x,line_2),cv2.FONT_HERSHEY_SIMPLEX, font_size, color_x, thickness)
		cv2.putText(frame, txt_y, (pos_x,line_3),cv2.FONT_HERSHEY_SIMPLEX, font_size, color_y, thickness)		

	else:

		cv2.putText(frame, "Searching board 7x7... (Cover ALL the circles)", (pos_x,line_1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), thickness)

	# Set up fullscreen display window
	cv2.namedWindow("Assisted Calibration Assistant", cv2.WINDOW_NORMAL)
	cv2.setWindowProperty("Assisted Calibration Assistant", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow("Assisted Calibration Assistant", frame)

	# Exit condition (Press 'q')
	if cv2.waitKey(1) & 0xFF == ord('q'):

		break

cap.release()
cv2.destroyAllWindows()
print("Calibration Finished")