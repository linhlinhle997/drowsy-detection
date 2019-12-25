"""
 USAGE:
        displayed to screen: python drowsy_eye.py 
        dont displayed to screen: python drowsy_eye.py -d 0
"""

import tensorflow as tf
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import dlib
import pyglet
import time
from threading import Thread
from collections import OrderedDict
from imutils.video import FPS
from pygame import mixer
from datetime import datetime
from time import gmtime, strftime
import os
from contextlib import contextmanager
import sys
from loadModel import loadModel, predictImage
from drawFace import predictFacialLandmark, drawEyes, eye_aspect_ratio
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed to screen")
args = vars(ap.parse_args())

COUNTER = 0
MAX_FRAME = 24
EYE_THRESH = 0.20
ALARM_ON = False

# Load model
model = loadModel('trained_model/model_1576224708.5281918.json', "trained_model/weight_1576224708.5281918.h5")

# Predict Facial Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('input/trained_data.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = WebcamVideoStream(src=0).start()
time.sleep(1.0)
fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h,w) = (None, None)

try: 
    # creating a folder named data 
    if not os.path.exists('output/output_text'): 
        os.makedirs('output/output_text') 
    # if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data')

sys.stdout=open('output/output_text/{}.txt'.format(str(datetime.strftime(datetime.now(), '%Y%m%d'))), "w")

# loop over frames from the video stream
counter = 0
while True:
    # it, and convert it to grayscale channels)
    frame = vs.read()
    frame = imutils.resize(frame, 500)
    rects = predictFacialLandmark(img=frame, detector=detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects_2 = detector(gray, 0)

    if len(rects)==0:
        text = "0 face found"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print(str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')), ' | ', '{}'.format(len(rects)), '| |')
    
    for rect in rects:
        counter += 1
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
            
        leftEyeRatio = shape[lStart:lEnd]
        leftEye = drawEyes(leftEyeRatio, frame)
        leftEAR = eye_aspect_ratio(leftEyeRatio)
        classLeft = predictImage(leftEye, model=model)

        rightEyeRatio = shape[rStart:rEnd]
        rightEye = drawEyes(rightEyeRatio, frame)
        rightEAR = eye_aspect_ratio(rightEyeRatio)
        classRight = predictImage(rightEye, model=model)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        #print(ear)

        #Draw line around eye
        # leftEyeHull = cv2.convexHull(leftEyeRatio)
        # rightEyeHull = cv2.convexHull(rightEyeRatio)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Draw face bounding box 
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (255, 255, 255), 1)

        #Draw current day, time
        cv2.putText(frame,str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)

        if len(rects) == 1 :
            text = "1 face found"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if classLeft == 0 and classRight == 0:
                # Eye is closing
                COUNTER += 1
                cv2.putText(frame, "Closing - NF: {:.2f}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')), " | 1 | Closing |")
                
                # Eye is Drowsy
                if COUNTER >= MAX_FRAME and ear <= EYE_THRESH: 
                    # check if the video writer is None
                    if writer is None:
                        save_dir = os.path.join('output/output_video/' + str(datetime.strftime(datetime.now(), '%Y%m%d')))
                        out_vid_path = os.path.join(save_dir + '/' + str(datetime.strftime(datetime.now(), '%Y%m%d_%H%M') + '.avi'))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        (h,w) = frame.shape[:2]
                        writer = cv2.VideoWriter(out_vid_path, fourcc, 5, (w,h),True)
                    
                    #If the alarm is not on, turn it on
                    # if not ALARM_ON:
                    #     ALARM_ON = True
                    #     mixer.init()
                    #     sound = mixer.Sound("input/alarm.wav")
                    #     sound.play()

                    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 3)
                    cv2.putText(frame, "DROWSY!", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                    print(str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')), ' | 1 | DROWSY | ', out_vid_path)
                    writer.write(frame)
                
            else:
                # Eye is Opening
                COUNTER = 0
                ALARM_ON = False
                writer = None
                cv2.putText(frame, "Opening", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                print(str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')), ' | 1 | Opening |' )
        
        elif len(rects) > 1:
            text = "{} faces found".format(len(rects))
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print(str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')), ' | ', '{}'.format(len(rects)), '| |')

    # check to see if we should display the output frame to our screen
    if args["display"] > 0:
        capname = "Frame"
        cv2.imshow(capname, frame)
        cv2.moveWindow(capname, 600, 80)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

sys.stdout.close()
vs.stream.release()
cv2.destroyAllWindows()
vs.stop()
