import cv2

# Open the first webcame device
capture = cv2.VideoCapture(

)
# capture = cv2.VideoCapture(0)

# Get the width and height of frame
# width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
width = 640
height = 480

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while True:
    ret, frame = capture.read()

    if ret is True:
        out.write(frame)

        cv2.imshow('frame', frame)

    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
