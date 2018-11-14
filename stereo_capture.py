import cv2

left_path = 'capture/left/{:06d}.jpg'
right_path = 'capture/right/{:06d}.jpg'

# Filenames are just an increasing number
frameId = 0

left_capture = cv2.VideoCapture('http://10.0.1.5:8081/img')
right_capture = cv2.VideoCapture('http://10.0.1.4:8081/img')

# Capture loop from earlier...
while(True):
    _, left_img = left_capture.read()
    _, right_img = right_capture.read()
    # Actually save the frames
    # cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
    # cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)
    frameId += 1

    cv2.imshow('left', left_img)
    cv2.imshow('right', right_img)

    cv2.imwrite(left_path.format(frameId), left_img)
    cv2.imwrite(right_path.format(frameId), right_img)
    frameId += 1

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break