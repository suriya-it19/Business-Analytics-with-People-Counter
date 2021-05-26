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

from mylib.table import table
from mylib import config, thread
from mylib.centroidtracker import CentroidTracker
from mylib.mailer import Mailer
from mylib.trackableobject import TrackableObject

input_videos = {'Dining':'videos/edit.mp4',
                'Dining1': 'video1.mp4',
                'Enterence': 'videos/example_01.mp4'
            }#config.url3

cost_price_model_path = 'models/cost_price/'
final_price_model_path = 'models/final_price/'
final_rating_model_path = 'models/final_rating/'
final_sold_model_path = 'models/final_sold/'
    
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
dish_price = ['chettinad_mutton_price', 'chicken_curry_price', 'chicken_nuggets_price', 'dhal_makni_price', 'mutton_chops_price', 'paneer_tikka_price', 'prawn_fry_price', 'veg_manchurian_price']
dish_rating = ['chettinad_mutton_rating', 'chicken_curry_rating', 'chicken_nuggets_rating', 'dhal_makni_rating', 'mutton_chops_rating', 'paneer_tikka_rating', 'prawn_fry_rating', 'veg_manchurian_rating']
dish_sold = ['chettinad_mutton_sold', 'chicken_curry_sold', 'chicken_nuggets_sold', 'dhal_makni_sold', 'mutton_chops_sold', 'paneer_tikka_sold', 'prawn_fry_sold', 'veg_manchurian_sold']

def run(area):
    o = 0
    t0 = time.time()

    df = pd.DataFrame({
                'Date': '0',
                'Week': '0',
                'Price': 0.0,
                'dishes': ['chettinad_mutton', 'chicken_curry', 'chicken_nuggets', 'dhal_makni', 'mutton_chops', 'paneer_tikka', 'prawn_fry', 'veg_manchurian'],
                'dish_sold' : [0,0,0,0,0,0,0,0],
                'dish_rating' : [0,0,0,0,0,0,0,0],
                'dish_price' : 0.0
            }).set_index('Date')

    bar1 = st.bar_chart(df[['dishes', 'dish_sold']].set_index('dishes'))
    bar2 = st.bar_chart(df[['dishes', 'dish_rating']].set_index('dishes'))
    line = st.line_chart(df[['Week', 'dish_price']].set_index('week'))

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(config.PROTOTXT, config.MODEL)

    with st.beta_container():
        c1,c2,c3 = st.beta_columns(3)
        with c1:
            image_placeholder1 = st.empty()
        with c2:
            image_placeholder2 = st.empty()
        with c3:
            image_placeholder3 = st.empty()

    if not input_videos:
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
        #vs = VideoStream(input_video).start()        
        vs = cv2.VideoCapture(input_videos[area])
        vs1 = cv2.VideoCapture(input_videos['Dining1'])
        vs2 = cv2.VideoCapture(input_videos['Enterence'])

    writer = None
    totalFrames = 0
    empty=[]
    empty1=[]
    x=[]
    status = 'Tracking'

    W = None
    H = None

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        if input_videos[area]:
            frame = frame[1] if vs else frame

        if config.url1 or config.url2 is None and frame is None:
            break

        frame = imutils.resize(frame, width = 700)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        status = 'Detecting'
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        totalIn = 0

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
                totalIn += 1
                #count = count + totalIn
                #if len(box):
                #    x += totalIn
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)               

        #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        #cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((0 * 20) + 200)),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        text = "In: {}".format(totalIn)
        cv2.putText(frame, text, (10, H - ((1 * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text = "Status: {}".format(status)
        cv2.putText(frame, text, (10, H - ((2 * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        #text = "Total people inside: {}".format(count)
        #cv2.putText(frame, text, (400, H - ((1 * 20) + 90)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue = '')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        image_placeholder2.image(frame, channels="BGR")
        
        success, frame1 = vs1.read()
        frame1 = imutils.resize(frame1, width = 700)
        image_placeholder1.image(frame1)
        
        success, frame2 = vs2.read()
        frame2 = imutils.resize(frame2, width = 700, height=300)
        image_placeholder3.image(frame2)
        o += 1

            
        if o % 20 ==0:
            upTable = table(df, totalIn, price, chettinad_mutton_plates, chettinad_mutton_rating, dish_sold, dish_rating, dish_price)
            bar1.add_rows(upTable[['dishes', 'dish_sold']].set_index('dishes'))
            bar2.add_rows(upTable[['dishes', 'dish_rating']].set_index('dishes'))
            line.add_rows(upTable[['week', 'dish_price']].set_index('week'))
