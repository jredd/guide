# Import the OpenCV library
import cv2
import dlib

# Initialize a face cascade using the frontal face haar cascade provided
# with the OpenCV2 library
personCascade = cv2.CascadeClassifier('/.jdev/python/.virtualenvs/guide/lib/python3.6/site-packages/cv2/data/haarcascade_fullbody.xml')
# personCascade = cv2.CascadeClassifier('case.xml')

# The desired output width and height
OUTPUT_SIZE_WIDTH = 640
OUTPUT_SIZE_HEIGHT = 480

# Create the tracker we will use
tracker = dlib.correlation_tracker()

# The variable we use to keep track of the fact whether we are
# currently using the dlib tracker
trackingPerson = 0

# Open the first webcame device
# capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture('man_walking.mp4')
capture = cv2.VideoCapture('output.mp4')
# capture = cv2.VideoCapture('http://10.0.1.4:8081/img')
while True:

    # Retrieve the latest image from the webcam
    rc, fullSizeBaseImage = capture.read()
    if not rc:
        print('Detection ended')
        break

    # Resize the image to 320x240
    baseImage = cv2.resize(fullSizeBaseImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

    resultImage = baseImage.copy()

    # If we are not tracking a face, then try to detect one
    if not trackingPerson:

        # For the face detection, we need to make use of a gray
        # colored image so we will convert the baseImage to a
        # gray-based image
        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        # Now use the haar cascade detector to find all faces
        # in the image
        persons = personCascade.detectMultiScale(gray, 1.01, 1)
        print(persons)
        # In the console we can show that only now we are
        # using the detector for a face
        print("Using the cascade detector to detect face")

        # For now, we are only interested in the 'largest'
        # face, and we determine this based on the largest
        # area of the found rectangle. First initialize the
        # required variables to 0
        maxArea = 0
        x = 0
        y = 0
        w = 0
        h = 0
        # print(faces)
        # Loop over all faces and check if the area for this
        # face is the largest so far
        # We need to convert it to int here because of the
        # requirement of the dlib tracker. If we omit the cast to
        # int here, you will get cast errors since the detector
        # returns numpy.int32 and the tracker requires an int
        for (_x, _y, _w, _h) in persons:
            if _w * _h > maxArea:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                maxArea = w * h

        # If one or more faces are found, initialize the tracker
        # on the largest face in the picture
        # if maxArea > 0:
        # Initialize the tracker
        tracker.start_track(
            baseImage,
            dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20)
        )

        # Set the indicator variable such that we know the
        # tracker is tracking a region in the image
        trackingPerson = 1

    # Check if the tracker is actively tracking a region in the image
    if trackingPerson:
        rectangleColor = (0, 165, 255)

        # Update the tracker and request information about the
        # quality of the tracking update
        trackingQuality = tracker.update(baseImage)

        # If the tracking quality is good enough, determine the
        # updated position of the tracked region and draw the
        # rectangle
        if trackingQuality >= 8.75:
            tracked_position = tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          rectangleColor, 2)

        else:
            # If the quality of the tracking update is not
            # sufficient (e.g. the tracked region moved out of the
            # screen) we stop the tracking of the face and in the
            # next loop we will find the largest face in the image
            # again
            trackingPerson = 0

        # # Create two opencv named windows
        # cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

        largeResult = cv2.resize(resultImage,
                                 (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

        # Finally, we want to show the images on the screen
        # cv2.imshow("base-image", baseImage)
        cv2.imshow("result-image", largeResult)

        # Position the windows next to eachother
        # cv2.moveWindow("base-image", 0,0)
        # cv2.moveWindow("result-image", 400, 100)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
