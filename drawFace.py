import imutils
import cv2
from scipy.spatial import distance as dist
import numpy as np

def predictFacialLandmark(img, detector):
    img = imutils.resize(img, 500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    return rects

def drawEyes(eye, image):
    (x, y, w, h) = cv2.boundingRect(np.array([eye]))
    h = w
    y = y - h // 2  
    roi = image[int(y):int(y + h), int(x):int(x + w)]
    roi = imutils.resize(roi, width=24, inter=cv2.INTER_CUBIC)

    return roi

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear