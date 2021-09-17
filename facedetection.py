import cv2
import mediapipe as mp
import time


# FACE DETECTION CLASS MODULAR

class FaceDetector:

    def __init__(self, detectionConfidence=0.5):

        self.detectionConfidence = detectionConfidence
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.detectionConfidence)  # initializing face detection, param is confidence =0.5

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(imgRGB)
        bboxes = []

        if self.results.detections:
            for id, my_detection in enumerate(self.results.detections):
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

                bboxes.append([id, bbox, my_detection.score])

                if draw:
                    # draws results
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f'{int(my_detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                    # printing results from json format:
                    # print(f'Face #{id}', " :::: \n" + str(bbox_class))

                    # draws points + box:
                    # mp_draw.draw_detection(img, my_detection)

        return bboxes, img

    def fancy_draw(self, img, bbox, l = 30, thick=7, rectthick=1):
        x, y, w, h = bbox
        xe, ye = x+w, y+h

        cv2.rectangle(img, bbox, (255, 255, 0), rectthick)
        # top left corner
        cv2.line(img, (x,y), (x+l, y), (255, 255, 0), thick)
        cv2.line(img, (x,y), (x, y+l), (255, 255, 0), thick)

        # bottom right corner
        cv2.line(img, (xe,ye), (xe-l, ye), (255, 255, 0), thick)
        cv2.line(img, (xe,ye), (xe, ye-l), (255, 255, 0), thick)

        # bottom left
        cv2.line(img, (x,ye), (x+l, ye), (255, 255, 0), thick)
        cv2.line(img, (x,ye), (x, ye-l), (255, 255, 0), thick)


        # top left corner
        cv2.line(img, (xe,y), (xe-l, y), (255, 255, 0), thick)
        cv2.line(img, (xe,y), (xe, y+l), (255, 255, 0), thick)

        return img



def write_fps(img, pTime):
    cTime = time.time()
    fps = 1/(cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return cTime

def main():
    path = "vids/stockvid.mp4"
    cap = cv2.VideoCapture(path)
    pTime = 0
    width, height = 960, 540
    detector = FaceDetector()

    while True:
        success, img = cap.read()

        # finding faces and returning the image
        bboxes, img = detector.findFaces(img)
        img = cv2.resize(img, (width, height))  # resize at end maybe?

        pTime = write_fps(img, pTime)


        cv2.imshow("Output", img)
        if cv2.waitKey(2) == ord('q'):
            # we wait (4) because video is loading too fast, if we process stuff then we can also delay it
            cv2.destroyAllWindows()
            break

if __name__ == "main":
    main()

# uncomment me to test this class "playing itself"
# otherwise keep it commented because if you import it ..
# ..while uncommented it will run a video each time
# main()