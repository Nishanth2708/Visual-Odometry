import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from ReadCameraModel import ReadCameraModel
from UndistortImage import *



fx,fy,cx,cy, G_camera_image, LUT = ReadCameraModel('./model')

path = r'C:\Users\12027\PyCharm_Projects\Visual_Odometry\Oxford_dataset\stereo\centre\*.png'

filenames = [img for img in glob.glob(path)]
# filenames = [img for img in glob.glob("/home/srujan/PycharmProjects/visual_odometry/samples/*.png")]

filenames.sort()
image_list =[]
for img in filenames:
    image_list.append(img)


def sift(img1, img2):
    img1 = img1
    img2 = img2

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()  # create sift features

    # Key Points and Descriptors for the corresponding images
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio_thresh * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(gray_img1, kp1, gray_img2, kp2, matches, None, **draw_params)

    #     show = plt.imshow( img3)

    return img3





