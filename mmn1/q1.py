import sys

import cv2
import numpy as np

from mmn1.mmn_utils import *


# section A:
# perform canny edge detection for input image
# threshold1= minimune threshold1 to be considared as an edge. anything below threshold1 would not be considared as edge.
# threshold2 = anything above threshold2 would be considared as edge for sure. anythong between threshold 1 and threshold 2 would be considared as an edge onlt if it's nighbours are past of an edge.
# returned as a binary image with canny edges
def canny_edge_detector(input_img, threshold1, threshold2, draw=True, save=True):
    canny_img = cv2.cvtColor(np.copy(input_img), cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', canny_img)
    cv2.waitKey(1)

    edges = cv2.Canny(canny_img, threshold1, threshold2)
    if draw:
        cv2.namedWindow('CannyEdges', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CannyEdges', 600, 600)
        cv2.imshow('CannyEdges', edges)
        cv2.waitKey(1)
    if save:
        cv2.imwrite('outputs//cannyout_edges.jpg', edges)

    return edges


# section B:
# perform Harries corners detection for input image and save the output, with corner markes, in a new file name "harries_coners_detection.jpg"
# blockSize=It is the size of neighbourhood considered for corner detection
# ksize=Aperture parameter of Sobel derivative used
# k=Harris detector free parameter in the equation.
def harries_conrners_detector(input_img, blockSize, ksize, k, draw=True, save=True):
    harriescrners_img = cv2.cvtColor(np.copy(input_img), cv2.COLOR_BGR2GRAY)
    gray = np.float32(harriescrners_img)
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)

    dst = cv2.dilate(dst, None)

    # Mark corner index pixels on gray image
    b, g, r = cv2.split(input_img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    rgb_img[dst > 0.01 * dst.max()] = [0, 0, 255]  # mark corner index pixels in red

    if draw:
        cv2.namedWindow('harries_corners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('harries_corners', 600, 600)
        cv2.imshow('harries_corners', rgb_img)
        cv2.waitKey(1)
    if save:
        cv2.imwrite('outputs//harriesout_corners.jpg', rgb_img)

    return rgb_img, dst


# section C:
# calc SIFT for each detected corner in an image.
# returned is SIFT keypoints(kp) and it's descriptors,des (vector size 128 for each point)
def sift(input_img, draw=True, save=True):
    sift_img = np.copy(input_img)
    gray = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # sift.detect finds the keypoint in the images.
    # Each keypoint is a special structure which has many attributes like its (x,y) coordinates, size of the meaningful neighbourhood,
    # angle which specifies its orientation, response that specifies strength of keypoints etc.
    kp, des = sift.detectAndCompute(gray, None)

    # cv.drawKeyPoints() function which draws the small circles on the locations of keypoints
    if draw:
        img = cv2.drawKeypoints(gray, kp, sift_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.namedWindow('sift_keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('sift_keypoints', 600, 600)
        cv2.imshow('sift_keypoints', sift_img)
        cv2.waitKey(1)
    if save:
        cv2.imwrite('outputs//siftout_keypoints.jpg', sift_img)

    return kp, des


# section D:
# find matching points on 2 input images
# 1. calc SIFT for each image
# 2. Law ratio between the images (sift distance in both images and ratio between the most closest point to the second closes point)
# take all matched points above decided threshold
# inputs are image1 and image2 and law ratio threshold
def matching_points(input_img1, input_img2, threshold=0.75, draw=True, save=True):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift(input_img1, draw=False, save=False)
    kp2, des2 = sift(input_img2, draw=False, save=False)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    if draw:
        img3 = None
        img3 = cv2.drawMatchesKnn(img1=input_img1, keypoints1=kp1, img2=input_img2, keypoints2=kp2, outImg=img3,
                                  matches1to2=good,
                                  flags=2)
        cv2.namedWindow('matched_keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('matched_keypoints', 600, 600)
        cv2.imshow('matched_keypoints', img3)
        cv2.waitKey(1)
    if save:
        cv2.imwrite('outputs//matched_keypoints.jpg', img3)

    return img3


# section E:
# perform hough transformation on an input image to find lines in the image
# perform GaussianBlur on image
# get binary image using canny edge detection
# use hough transformation to detect lines using:
# threshold= Accumulator threshold parameter. Only those lines are returned that get enough
# minLineLength= Minimum line length. Line segments shorter than that are rejected.
# axLineGap= Maximum allowed gap between points on the same line to link them.
def hough_transform(input_img, draw=True, save=True):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # find canny edges
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 50, 150)
    cv2.imwrite('outputs//canny_withblur.jpg', edges)

    # find hough lines using found edges
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=100, maxLineGap=10)
    # top_lines = lines[0:100]

    if draw:
        hough_lines = input_img
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.namedWindow('hough_lines', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('hough_lines', 600, 600)
        cv2.imshow('hough_lines', hough_lines)
        cv2.waitKey(1)

        if save:
            cv2.imwrite('outputs//houghlines.jpg', hough_lines)


if __name__ == '__main__':
    # read input image file path from user and section to be executed
    arg_dict = command_line_args(argv=sys.argv)
    if "image1_path" in (arg_dict.keys()):
        image1_path = str(arg_dict.get('image1_path')[0])
    else:
        image1_path = "inputs//gan1.jpg"
    if "image2_path" in (arg_dict.keys()):
        image2_path = str(arg_dict.get('image2_path')[0])
    else:
        image2_path = "inputs//gan2.jpg"
    if "section" in (arg_dict.keys()):
        section = str(arg_dict.get('section')[0])
    else:
        section = "A"

    img1 = cv2.imread(image1_path)

    # A: canny edge detection
    if section == "A":
        print("A: canny edge detection")
        canny_edges = canny_edge_detector(img1, 50, 100)  # threshold1, threshold2

    # B: Harries corners detection
    # inputs are blockSize, ksize,and k
    if section == "B":
        print("B: Harries corners detection")
        rgb_img, dst = harries_conrners_detector(img1, 2, 3, 0.04)

    # C: SIFT keypoint and descriptors detection
    if section == "C":
        print("C: SIFT keypoint and descriptors detection")
        kp, des = sift(img1)

    # D: Matching interest points
    # 1. calc SIFT for each image
    # 2. Law ratio between the images (sift distance in both images and ratio between the most closest point to the second closes point)
    # take all matched points above decided threshold
    # inputs are image1 and image2 and law ratio threshold
    if section == "D":
        print("D: Matching interest points between two images")
        img2 = cv2.imread(image2_path)
        matching_points(img1, img2, 0.5)

    # E: Hough line in image
    # 1. get image with edges, using canny edges detector
    # 2. find lines using hough lines detector
    if section == "E":
        print("E: Hough line in image")
        hough_transform(img1)
