import cv2
import os

# Define gesture name and number of images to capture
gesture_name = 'mano-Philippines'  # Update for each gesture
num_images = 50
save_dir = f'datasets/{gesture_name}'

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize camera
cap = cv2.VideoCapture(0)
count = 0

print(f"Capturing images for '{gesture_name}' gesture")

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    # Display and capture frame
    cv2.imshow("Capture Gesture", frame)
    img_name = os.path.join(save_dir, f"{gesture_name}_{count}.jpg")
    cv2.imwrite(img_name, frame)
    count += 1
    print(f"Captured image {count}/{num_images}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Image capture complete.")
