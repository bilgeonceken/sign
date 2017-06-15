###  THE CODE I RIPPED THIS IS FOR PYTHON 2.7  ###
###  I HOWEVER, USE PYTHON 3.5 -- JUST SAYIN'  ###

import numpy as np
from PIL import ImageGrab
import cv2

from ourfuncs import (auto_canny, four_point_transform, order_points,
                      resize_and_threshold_warped)


min_area = 5000
min_diff = 10000

## A4 Paper is 297cm*210cm
## we can multiply this values with a constant(pcon)
## to well... See better on the screen?
pcon = 3
w=297*pcon
h=210*pcon

# h=595
# w=842


class Sign(object):
    """string name, string destination"""
    def __init__(self, name, file):
        self.name = name
        im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.img = cv2.resize(imgray,(w//2,h//2),interpolation = cv2.INTER_AREA)

l_arrow = Sign("Left Arrow", "ArrowL - READY.jpg")


## image to be inspected
wr = cv2.imread("raw.jpg", cv2.IMREAD_UNCHANGED)


gray = cv2.cvtColor(wr, cv2.COLOR_BGR2GRAY)
cv2.imshow("wr", gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow("wr", blurred)
cv2.waitKey(0)

edges = auto_canny(blurred)
cv2.imshow("wr", edges)
cv2.waitKey(0)

cntr_frame, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("wr", cntr_frame)
cv2.waitKey(0)

warped_eq = None

"""
    For every contour in contours first we check whether its length is 4 or not
    Since every rectangle(ish) shape has four corners.

    if true  we check if its area is matches our criteria,
    if that is also true, we take thak rectangle from original image
    apply perpective correction and compare it with referance image
    using XOR function.

    if unmatched pixel count is less than what we want than it is a match
"""
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx) == 4:
        area = cv2.contourArea(approx)

        if area > min_area:
            cv2.drawContours(wr,[approx],0,(0,0,255),1)
            warped = four_point_transform(wr, approx.reshape(4,2), w, h)
            warped_eq = resize_and_threshold_warped(warped)
            diff_img = cv2.bitwise_xor(warped_eq, l_arrow.img)
            diff = cv2.countNonZero(diff_img)

            ## Checks if matches
            if diff < min_diff:
                # match = True
                cv2.putText(wr,l_arrow.name, tuple(approx.reshape(4,2)[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
                min_area
                break

if warped_eq.all():
    cv2.imshow("real_sign_data", l_arrow.img)
    cv2.imshow("warped_input",warped_eq)
    cv2.imshow("bitwise", diff_img)
    cv2.imshow("wr", wr)
    cv2.waitKey(0)
else:
    print("found no shit")



"""
    This is the video version of what is going on up there.
    Takes frames from video input and that do what its gonna to for each frame
"""

# video = cv2.VideoCapture(0)
# while(True):


#     printscreen_pil =  ImageGrab.grab(bbox = (0,100,640,480))
#     printscreen_numpy =   np.array(printscreen_pil, dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))

#     ret, printscreen_numpy = video.read()

#     gray = cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray,(3,3),0)

#     edges = auto_canny(blurred)


#     cntr_frame, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#         if len(approx) == 4:
#             area = cv2.contourArea(approx)

#             if area > min_area:
#                 cv2.drawContours(printscreen_numpy,[approx],0,(0,0,255),2)
#                 warped = four_point_transform(printscreen_numpy, approx.reshape(4,2), w, h)
#                 cv2.imshow("Corrected Perspective", warped_eq)
#                 cv2.imshow("Matching Operation", diffImg)
#                 cv2.imshow("Contours", edges)

#     cv2.imshow('window', printscreen_numpy)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         video.release()
#         cv2.destroyAllWindows()
#         break

