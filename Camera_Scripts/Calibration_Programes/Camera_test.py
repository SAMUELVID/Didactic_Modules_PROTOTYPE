import cv2
import time

# --- 1. INITIALIZATION ---
print("Initzializing camera test...")

# Open default camera (index 0)
cap = cv2.VideoCapture(0)

# Allow camera sensor to warm up
time.sleep(1)

# Check if the camera successfully opened
if not cap.isOpened():

	print("FATAL Error: Python can not connect to the camera")

else:
	
	print("Camera connected pres q to exit.")

	# --- 2. MAIN LOOP ---
	while True:

		ret,frame = cap.read()

		# Check for signal loss
		if not ret:

			print("Signal lost")

			break

		# Flip frame horizontally for a mirror effect
		frame = cv2.flip(frame,1)

		# Display the live feed
		cv2.imshow(" Pure Camera Test ",frame)

		# Exit condition (Press 'q')
		if cv2.waitKey(1) & 0xFF == ord('q'):

			break

# --- 3. CLEANUP ---
cap.release()
cv2.destroyAllWindows
print("Finalized test")