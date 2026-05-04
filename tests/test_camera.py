import cv2
import time

for i in range(6):
    print(f"\n=== Testing camera index {i} ===")
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print(f"Camera index {i}: opened=False")
        cap.release()
        continue

    print(f"Camera index {i}: opened=True")
    time.sleep(1.0)

    ret, frame = cap.read()
    print(f"  read={ret}, shape={None if frame is None else frame.shape}")

    if ret and frame is not None:
        cv2.imshow(f"cam_{i}", frame)
        cv2.waitKey(1000)
        cv2.destroyWindow(f"cam_{i}")

    cap.release()

cv2.destroyAllWindows()