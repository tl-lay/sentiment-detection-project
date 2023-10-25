import os
import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import pandas as pd
import csv
from collections import Counter

# load model
model = load_model("model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self):

        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.releast()

    def get_frame(self):
        ret, frame = self.video.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        with open('predictions.csv', 'a') as f:
            writerObj = csv.writer(f)
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                # find max indexed array
                max_index = np.argmax(predictions[0])
                emotions = ('negative', 'neutral', 'positive')
                writerObj.writerow([emotions[max_index]])
                cv2.putText(frame, emotions[max_index], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(frame, (1000, 700))
        ret, jpeg = cv2.imencode('.jpg', resized_img)
        return jpeg.tobytes()