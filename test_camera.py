import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Camera index {i}: opened={ok}")
    if ok:
        ret, frame = cap.read()
        print(f"  read={ret}, shape={None if frame is None else frame.shape}")
        cv2.imshow(f"cam_{i}", frame if ret else cv2.imread(""))
        cv2.waitKey(1000)
    cap.release()

cv2.destroyAllWindows()