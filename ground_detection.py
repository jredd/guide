# import the necessary packages
import numpy as np
import argparse
# import imutils
import time
from matplotlib import pyplot as plt

import cv2

def run():

    while True:
        r, img = cap.read()

        if not r:
            break

        edges = cv2.Canny(img, 100, 200)

        # plt.subplot(121), plt.imshow(img, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()

        cv2.imshow("preqview", img)
        cv2.imshow("edges", edges)

        # kernel = np.ones((5, 5), np.uint8)
        # gauss = cv2.GaussianBlur(img, (3, 3), 0)
        # grey = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel=kernel)
        # dilate = cv2.dilate(morph, kernel=kernel, iterations=1)
        # # water = cv2.watershed(dilate)
        # cv2.imshow("preqview", img)
        # cv2.imshow("gaussian", gauss)
        # cv2.imshow("grey", grey)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("morph", morph)
        # cv2.imshow("dilate", dilate)
        #
        # im2, contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # # cnt = contours[8]
        # # cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
        #
        # cv2.imshow("img", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    cap = cv2.VideoCapture('http://10.0.1.5:8081/img')

    run()
