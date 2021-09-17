from facedetection import FaceDetector
import cv2
import time

path = 0
cap = cv2.VideoCapture(0)
pTime = 0
# width, height = 960, 540
detector = FaceDetector()

def write_fps(img, pTime):
    cTime = time.time()
    fps = 1/(cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return cTime

while True:
    success, img = cap.read()

    # finding faces and returning the image
    bboxes, img = detector.findFaces(img)
    # img = cv2.resize(img, (width, height))  # resize at end maybe?

    pTime = write_fps(img, pTime)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord('q'):
        # we wait (4) because video is loading too fast, if we process stuff then we can also delay it
        cv2.destroyAllWindows()
        break