import argparse
import csv
import datetime
import time
from itertools import zip_longest
from random import randrange

import cv2
import dlib
import imutils
import joblib
import numpy as np
import pandas as pd
import schedule
import streamlit as st
from imutils.video import FPS, VideoStream

from mylib import config, thread
from mylib.centroidtracker import CentroidTracker
from mylib.mailer import Mailer
from mylib.trackableobject import TrackableObject

input_video = 'videos/example_01.mp4'#config.url3

cost_price_model_path = 'F:/TCS_Inframind/Restaurant/models/cost_price/'
final_price_model_path = 'F:/TCS_Inframind/Restaurant/models/final_price/'
final_rating_model_path = 'F:/TCS_Inframind/Restaurant/models/final_rating/'
final_sold_model_path = 'F:/TCS_Inframind/Restaurant/models/final_sold/'
    
price = joblib.load(cost_price_model_path + 'price.pkl')

chettinad_mutton_price = joblib.load(final_price_model_path + 'chettinad_mutton_price.pkl')
chicken_curry_price = joblib.load(final_price_model_path + 'chicken_curry_price .pkl')
chicken_nuggets_price = joblib.load(final_price_model_path + 'chicken_nuggets_price.pkl')
dhal_makni_price = joblib.load(final_price_model_path + 'dhal_makni_price.pkl')
mutton_chops_price = joblib.load(final_price_model_path + 'mutton_chops_price.pkl')
paneer_tikka_price = joblib.load(final_price_model_path + 'paneer_tikka_price.pkl')
prawn_fry_price = joblib.load(final_price_model_path + 'prawn_fry_price.pkl')
veg_manchurian_price = joblib.load(final_price_model_path + 'veg_manchurian_price.pkl')

chettinad_mutton_rating = joblib.load(final_rating_model_path + 'chettinad_mutton_rating.pkl')
chicken_curry_rating = joblib.load(final_rating_model_path + 'chicken_curry_rating.pkl')
chicken_nuggets_rating = joblib.load(final_rating_model_path + 'chicken_nuggets_rating.pkl')
dhal_makni_rating = joblib.load(final_rating_model_path + 'dhal_makni_rating.pkl')
mutton_chops_rating = joblib.load(final_rating_model_path + 'mutton_chops_rating.pkl')
paneer_tikka_rating = joblib.load(final_rating_model_path + 'paneer_tikka_rating.pkl')
prawn_fry_rating = joblib.load(final_rating_model_path + 'prawn_fry_rating.pkl')
veg_manchurian_rating = joblib.load(final_rating_model_path + 'veg_manchurian_rating.pkl')

chettinad_mutton_plates = joblib.load(final_sold_model_path + 'chettinad_mutton_plates.pkl')
chicken_curry_plates = joblib.load(final_sold_model_path + 'chicken_curry_plates.pkl')
chicken_nuggets_plates = joblib.load(final_sold_model_path + 'chicken_nuggets_plates.pkl')
dhal_makni_plates = joblib.load(final_sold_model_path + 'dhal_makni_plates.pkl')
mutton_chops_plates = joblib.load(final_sold_model_path + 'mutton_chops_plates.pkl')
paneer_tikka_plates = joblib.load(final_sold_model_path + 'paneer_tikka_plates.pkl')
prawn_fry_plates = joblib.load(final_sold_model_path + 'prawn_fry_plates.pkl')
veg_manchurian_plates = joblib.load(final_sold_model_path + 'veg_manchurian_plates.pkl')

def run():
    t0 = time.time()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(config.PROTOTXT, config.MODEL)
    if not input_video:
        if config.url2:
            print("[INFO] Starting the live stream..")
            """
            [INFO] Starting the live stream..
            """
            vs = VideoStream(config.url2).start()
            time.sleep(2.0)
        else:
            print("[INFO] Starting the live stream..")
            """
            [INFO] Starting the live stream..
            """
            vs = VideoStream(config.url1).start()
            time.sleep(2.0)

    else:
        print("[INFO] Starting the video..")
        """
        [INFO] Starting the video..
        """
        vs = VideoStream(input_video).start()        
        #vs = cv2.VideoCapture(input_video)

    image_placeholder = st.empty()

    writer = None
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    W = None
    H = None

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        if input_video:
            frame = frame[1] if vs else frame

        if config.url1 or config.url2 is None and frame is None:
            break

        frame = imutils.resize(frame, width = 500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > config.Confidence:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")     

                # display the prediction
                COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                #print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)               

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        image_placeholder.image(frame, channels="BGR")

        def table():
            week = 1
            month = 1
            year = 5#randrange(5)
            date = f'{month}/2021'

            df = pd.DataFrame({
                'Date': f'{date}',
                'Week': f'{week}',
                'Price': price.predict([[year, month, week]]),
                'chettinad_mutton_plates' : chettinad_mutton_plates.predict([[year, month, week]]),
                'chettinad_mutton_rating' : chettinad_mutton_rating.predict([[year, month, week]]),
                'chettinad_mutton_price' : chettinad_mutton_price.predict([[year, month, week]])
            }, index=[0])

            st.table(df)#,width = 10000)
        
        table()