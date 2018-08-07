# Import the OpenCV library
import cv2
import dlib

# Initialize a face cascade using the frontal face haar cascade provided
# with the OpenCV2 library
faceCascade = cv2.CascadeClassifier('/.jdev/python/.virtualenvs/guide/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# The deisred output width and height
OUTPUT_SIZE_WIDTH = 640
OUTPUT_SIZE_HEIGHT = 480

# Open the first webcame device
capture = cv2.VideoCapture('http://10.0.1.3:8081/img')

while True:

    # Retrieve the latest image from the webcam
    rc, fullSizeBaseImage = capture.read()

    # Resize the image to 320x240
    baseImage = cv2.resize(fullSizeBaseImage, (320, 240))

    # # Check if a key was pressed and if it was Q, then destroy all
    # # opencv windows and exit the application
    # pressedKey = cv2.waitKey(2)
    # if pressedKey == ord('Q'):
    #     cv2.destroyAllWindows()
    # exit(0)

    # Result image is the image we will show the user, which is a
    # combination of the original image from the webcam and the
    # overlayed rectangle for the largest face
    resultImage = baseImage.copy()

    # For the face detection, we need to make use of a gray colored
    # image so we will convert the baseImage to a gray-based image
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    # Now use the haar cascade detector to find all faces in the
    # image
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    # For now, we are only interested in the 'largest' face, and we
    # determine this based on the largest area of the found
    # rectangle. First initialize the required variables to 0
    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0
    rectangleColor = (0, 165, 255)

    # Loop over all faces and check if the area for this face is
    # the largest so far
    for (_x, _y, _w, _h) in faces:
        if _w * _h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w * h

        # If one or more faces are found, draw a rectangle around the
        # largest face present in the picture
        if maxArea > 0:
            cv2.rectangle(resultImage, (x - 10, y - 20),
                          (x + w + 10, y + h + 20),
                          rectangleColor, 2)

    # Since we want to show something larger on the screen than the
    # original 320x240, we resize the image again
    #
    # Note that it would also be possible to keep the large version
    # of the baseimage and make the result image a copy of this large
    # base image and use the scaling factor to draw the rectangle
    # at the right coordinates.
    largeResult = cv2.resize(resultImage,
                             (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

    # Finally, we want to show the images on the screen
    # cv2.imshow("base-image", baseImage)
    cv2.imshow("result-image", largeResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
