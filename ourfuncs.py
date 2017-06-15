import numpy as np
import cv2

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, width, heigth):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    maxWidth = width//2
    maxHeight = heigth//2

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def resize_and_threshold_warped(image):
    #Resize the corrected image to proper size & convert it to grayscale
    #warped_new =  cv2.resize(image,(w/2, h/2))
    warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Smoothing Out Image
    blur = cv2.GaussianBlur(warped_new_gray,(5,5),0)

    #Calculate the maximum pixel and minimum pixel value & compute threshold
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
    threshold = (min_val + max_val)/2

    #Threshold the image
    ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)

    #return the thresholded image
    return warped_processed

def bin_giver(img):
    """
        Takes an image, returns a version of it
        where only white is white and rest is black
    """
    upper = np.array([255,255,255])
    lower = np.array([133,133,133])

    mask = cv2.inRange(img, lower, upper)
    # rev_mask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(img,img, mask= mask)
    # rev_res = cv2.bitwise_and(img,img, mask= rev_mask)