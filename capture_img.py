import cv2
import time
path = 'capture/{:06d}.jpg'

# Filenames are just an increasing number
frameId = 0

# capture = cv2.VideoCapture('http://10.0.1.4:8081/img')
capture = cv2.VideoCapture('http://10.0.1.5:8081/img')

# Capture loop from earlier...
while(True):
    _, img = capture.read()
    # Actually save the frames
    # cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
    # cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)
    frameId += 1

    cv2.imshow('img', img)

    cv2.imwrite(path.format(frameId), img)
    frameId += 1

    time.sleep(1)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break