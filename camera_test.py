import cv2

print("Testing cameras...")
print("Press Q to close the current camera.")
print("Press N to move to the next camera.")

for camera_index in range(5):

    print(f"\nTrying camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera {camera_index}: not available")
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = f"Camera {camera_index}"

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Camera {camera_index}: no frame")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            print("Test stopped.")
            exit()

        if key == ord("n"):
            break

    cap.release()
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

print("\nNo more cameras found.")
cv2.destroyAllWindows()