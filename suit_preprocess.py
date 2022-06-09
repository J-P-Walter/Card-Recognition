import cv2
import os

# Walks through the suit images, converts to gray and resizes, not sure if resize needed, might think about blurring
# if doing feature detection because it looked like it detected features of the texture of the cards
# Might also try to get better images for the templates
def process():
    for root, dirs, files in os.walk("suits"):
        for pic in files:
            suit = cv2.imread("suits" + "\\" + pic)  # INPUT
            cv2.imshow('a', suit)
            cv2.waitKey(0)
            gray = cv2.cvtColor(suit, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 240))
            cv2.imshow('a', gray)
            cv2.waitKey(0)
            #blur = cv2.GaussianBlur(gray, (5, 5), 0)
            #retval, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
            #cv2.imshow('a', thresh)
            #cv2.waitKey(0)
            cv2.imwrite("suits" + "\\" + pic, gray)
    print("suits processed")

