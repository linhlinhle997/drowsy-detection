from keras.preprocessing import image as keras_image
from keras.models import model_from_json
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import dlib
import pyglet
import time
from scipy.spatial import distance as dist
import playsound
from collections import OrderedDict
from imutils.video import FPS
import argparse

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
    classes = model.predict_classes(images, batch_size=10)
    return classes[0][0]

def predictFacialLandmark(img, detector):
    img = imutils.resize(img, width=500)
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

if __name__ == "__main__":
    # Define counter
    COUNTER = 0
    MAX_FRAME = 36
    
    # Load model
    model = loadModel('trained_model/model_1576057050.8468416.json', "trained_model/weight_1576057050.8468416.h5")

    # Predict Facial Landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('trained_data.dat')

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()

# loop over frames from the video stream
    counter = 0
    while True:
        # it, and convert it to grayscale channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        rects = predictFacialLandmark(img=frame, detector=detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# loop over the face detections
        for rect in rects:
            counter += 1

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEyeRatio = shape[lStart:lEnd]
            leftEye = drawEyes(leftEyeRatio, frame)
            classLeft = predictImage(leftEye, model=model)

            rightEyeRatio = shape[rStart:rEnd]
            rightEye = drawEyes(rightEyeRatio, frame)
            classRight = predictImage(rightEye, model=model)
            
            leftEyeHull = cv2.convexHull(leftEyeRatio)
            rightEyeHull = cv2.convexHull(rightEyeRatio)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (255, 255, 255), 1)

            if classLeft == 0 and classRight == 0:
                COUNTER += 1
                cv2.putText(frame, "Closing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "NF: {:.2f}".format(COUNTER), (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                print('Closing - ', (classLeft, classRight))

                if COUNTER >= MAX_FRAME:
                    cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 1)
                    cv2.putText(frame, "DROWSY!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print('DROWSY - ', (classLeft, classRight))
                    print("NF: {:.2f}".format(COUNTER))
            else:
                COUNTER = 0
                cv2.putText(frame, "Opening", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                print('Opening - ', (classLeft, classRight))
                

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()

        if key == ord("q"):
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

