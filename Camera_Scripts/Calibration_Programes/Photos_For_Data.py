import cv2
import os

# 1. Create the folder to save the photos if it does not exist
folder = 'calibration_photos'

if not os.path.exists(folder):
    
    os.makedirs(folder)
    print(f"Folder '{folder}' created.")

# 2. Initialize the camera
# Note: The index '0' usually points to the default integrated camera. 
# If you are on a laptop and have connected the robot's camera via USB, 
# it might be assigned to index '1' or '2'. Change this number if necessary.
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow("Camera Robot - Photo for Calibration", cv2.WINDOW_NORMAL)

img_counter = 0

print("\n--- INSTRUCTIONS ---")
print("1. Move the pattern over the image.")
print("2. Press SPACE BAR to take a photo.")
print("3. Press 'ESC' key to exit the program.")
print("---------------------\n")

while True:

    ret, frame = cam.read()
    
    if not ret:

        print("Error accessing the camera.")
    
        break

    # Display a brief help text overlay on the live video feed
    cv2.putText(frame, "Space Bar: Photo | ESC: Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Photos taken: {img_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera Robot - Photo for Calibration", frame)

    # Wait for a keystroke (1 ms delay)
    k = cv2.waitKey(1)

    if k % 256 == 27:

        # ESC key has been pressed
        print("Closing the photo taker...")
        
        break
    
    elif k % 256 == 32:
        
        # SPACE BAR has been pressed
        # Generate the filename (e.g., Photo_0.jpg, Photo_1.jpg)
        img_name = os.path.join(folder, f"Photo_{img_counter}.jpg")
        
        # Read a fresh, clean frame from the buffer without the green text overlay to save it
        clean_ret, clean_frame = cam.read()
        
        cv2.imwrite(img_name, clean_frame)
        print(f"Photo saved! -> {img_name}")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()