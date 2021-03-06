import numpy as np
import cv2
from imutils.object_detection import non_max_suppression


hog = cv2.HOGDescriptor()
# hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# cap = cv2.VideoCapture('man_walking.mp4')
cap = cv2.VideoCapture('output.mp4')
# cap = cv2.VideoCapture('http://10.0.1.5:8081/img')


while True:
    _, frame = cap.read()
    # baseImage = cv2.resize(frame, (320, 240))
    resultImage = frame.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(
        resultImage,
        winStride=(4, 4),
        # padding=(32, 32),
        padding=(8, 8),
        scale=1.1)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(resultImage, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    # cv2.imshow("Before NMS", baseImage)
    cv2.imshow("After NMS", resultImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
