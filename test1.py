from FaceDetectionModule import FaceDetector
import cv2
import pafy
import cvzone

url='https://www.youtube.com/watch?v=Da41FWyAklE'
videoPafy = pafy.new(url)
best = videoPafy.getbest()
fpsReader = cvzone.FPS()
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(best.url)
cap.set(3, 1280)
cap.set(4, 720)
detector = FaceDetector()
while True:
    success, img = cap.read()
    # img = cv2.resize(img,(1024,768))
    img, bboxs = detector.findFaces(img) ###########
    detector = FaceDetector(modelSelection=1,minDetectionCon=0.3)
    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        print(bboxs)
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    fps, img = fpsReader.update(img,pos=(50,80),color=(0,255,0),scale=5,thickness=5)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()