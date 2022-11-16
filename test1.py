from FaceDetectionModule import FaceDetector
import cv2

# cap = cv2.VideoCapture(0)
detector = FaceDetector(modelSelection=1,minDetectionCon=0.3)
img = cv2.imread('./aaa.jpg')
img, bboxs = detector.findFaces(img) ###########
while True:
    # success, img2 = cap.read()

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        print(bboxs)
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cap.release()
cv2.destroyAllWindows()