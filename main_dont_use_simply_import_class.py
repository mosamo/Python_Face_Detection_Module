import cv2
import mediapipe as mp
import time

path = "vids/stockvid.mp4"
pTime = 0
width, height = 960, 540

cap = cv2.VideoCapture(path)

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection() # initializing face detection, param is confidence =0.5

def write_fps(img, pTime):
    cTime = time.time()
    fps = 1/(cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return cTime

while True:
    success, img = cap.read()

    img = cv2.resize(img, (width, height)) # better resize at the end?

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)

    if results.detections:
        for id, my_detection in enumerate(results.detections):
            # enumerate makes the loop return an iteration number (loop 1 = 0, loop 2 = 1.. )
            # useful if you want to use "for obj in object" but also want to create an id

            # IMPORTANT, Height Width channels are flipped
            h, w, channels = img.shape

            bbox_class = my_detection.location_data.relative_bounding_box
            # xmin, ymin, and other relative bounding box attributes are returned in
            # in normalized format (0~1, 1 being max/far value),
            # to get accurate pixel measurements, we multiply it by width/height and use int() since it returns a float
            bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), \
                   int(bbox_class.width * w), int(bbox_class.height * h)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(my_detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # printing results from json format
            print(f'Face #{id}', " :::: \n" + str(bbox_class))

            # draws points + box: mp_draw.draw_detection(img, my_detection)



    pTime = write_fps(img, pTime)


    cv2.imshow("Output", img)
    if cv2.waitKey(2) == ord('q'):
        # we wait (4) because video is loading too fast, if we process stuff then we can also delay it
        break

cv2.destroyAllWindows()