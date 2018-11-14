import numpy as np
import cv2


def unsharp_mask(img, blur_size = (9,9), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)

def run():

    while True:
        r, img = cap.read()

        if not r:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret, thresh = cv2.threshold(blurred, 127, 255, 1)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        _, contours, h = cv2.findContours(thresh, 1, 2)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            print(len(approx))
            if len(approx) == 5:
                print("pentagon")
                cv2.drawContours(img, [cnt], 0, 255, -1)
            elif len(approx) == 3:
                print("triangle")
                cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
            elif len(approx) == 4:
                print("square")
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
            elif len(approx) == 9:
                print("half-circle")
                cv2.drawContours(img, [cnt], 0, (255, 255, 0), -1)
            elif len(approx) > 15:
                print("circle")
                cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)

        cv2.imshow('img', img)
        cv2.imshow('thresh', thresh)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    cap = cv2.VideoCapture('http://10.0.1.5:8081/img')

    run()
