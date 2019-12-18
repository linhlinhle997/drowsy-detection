"""
 USAGE:
        python drowsy_eye.py -a alarm.wav
"""
import tensorflow as tf
from keras.preprocessing import image as keras_image
from keras.models import model_from_json 
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import dlib
import pyglet
import time
from threading import Thread
from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.video import FPS
from pygame import mixer
from datetime import datetime
from time import gmtime, strftime
import os

def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def predictImage(img, model):
    img = np.dot(np.array(img, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=30)
    return classes[0][0]

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

if __name__ == "__main__":
    
    COUNTER = 0
    MAX_FRAME = 24
    EYE_THRESH = 0.20
    ALARM_ON = False
    START_T = 0
    start_time = 0
    end_time = 0

    # Load model
    model = loadModel('trained_model/model_1576224708.5281918.json', "trained_model/weight_1576224708.5281918.h5")

    # Predict Facial Landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('trained_data.dat')

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    (h,w) = (None, None)


# loop over frames from the video stream
    counter = 0
    while True:
        # it, and convert it to grayscale channels)
        frame = vs.read()
        frame = imutils.resize(frame, 500)

        rects = predictFacialLandmark(img=frame, detector=detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects_2 = detector(gray, 0)

        
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

            if len(rects) > 0:
                text = "{} face(s) found".format(len(rects))
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            #Draw face bounding box 
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (255, 255, 255), 1)
            #Draw current day, time
            cv2.putText(frame,str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)


            if classLeft == 0 and classRight == 0:
                # Eye is closing
                COUNTER += 1
                cv2.putText(frame, "Closing - NF: {:.2f}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print('Closing - NF: {:.2f}'.format(COUNTER), " - Datetime: ", str(datetime.strftime(datetime.now(), '%Y/%m/%d %Hh:%Mm:%Ss')))
                # Eye is Drowsy
                if COUNTER >= MAX_FRAME and ear <= EYE_THRESH: 
                    
                    if writer is None:
                        save_dir = os.path.join('output_video/' + str(datetime.strftime(datetime.now(), '%Y%m%d')))
                        out_vid_path = os.path.join(save_dir + '/' + str(datetime.strftime(datetime.now(), '%Y%m%d_%H%M') + '.mp4'))

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        (h,w) = frame.shape[:2]
                        writer = cv2.VideoWriter(out_vid_path, fourcc, 5, (w,h),True)
                    
                    #If the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        mixer.init()
                        sound = mixer.Sound("alarm.wav")
                        sound.play()

                    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 3)
                    cv2.putText(frame, "DROWSY!", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                    print('DROWSY - Datetime: ', str(datetime.strftime(datetime.now(), '%Y/%m/%d %Hh:%Mm:%Ss')))

                    writer.write(frame)
                
            else:
                # Eye is Opening
                COUNTER = 0
                ALARM_ON = False
                writer = None
                cv2.putText(frame, "Opening", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                print('Opening - Datetime: ', str(datetime.strftime(datetime.now(), '%Y/%m/%d %Hh:%Mm:%Ss')))

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

vs.stream.release()
cv2.destroyAllWindows()
vs.stop()
