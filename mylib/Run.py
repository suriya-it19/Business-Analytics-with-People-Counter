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

input_video = ''

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
        print("[INFO] Starting the live stream..")
        """
        [INFO] Starting the live stream..
        """
        vs = VideoStream(config.url).start()
        time.sleep(2.0)

    else:
        print("[INFO] Starting the video..")
        """
        [INFO] Starting the video..
        """
        vs = cv2.VideoCapture(input_video)

    image_placeholder = st.empty()

    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty=[]
    empty1=[]

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        if input_video:
            frame = frame[1] if vs else frame

        if config.url is None and frame is None:
            break

        frame = imutils.resize(frame, width = 500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % config.Skip_frames == 0:
            status = "Detecting"
            trackers = []

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

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

        else:
            for tracker in trackers:
                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        empty.append(totalUp)
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        empty1.append(totalDown)
                        x = []
                        x.append(len(empty1)-len(empty))
                        if sum(x) >= config.Threshold:
                            cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                print("[INFO] Sending email alert..")
                                Mailer().send(config.MAIL)
                                print("[INFO] Alert sent")

                        to.counted = True


            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ("Status", status),
        ]

        info2 = [
        ("Total people inside", x),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue = '')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        image_placeholder.image(frame, channels="BGR")

        def table():
            week = 1
            month = 1
            year = randrange(5)
            date = f'{month}/2021'

            df = pd.DataFrame({
                'Date': f'{date}',
                'Week': f'{week}',
                'Price': price.predict([[year, month, week]]),
                'chettinad_mutton_plates' : chettinad_mutton_plates.predict([[year, month, week]]),
                'chettinad_mutton_rating' : chettinad_mutton_rating.predict([[year, month, week]]),
                'chettinad_mutton_price' : chettinad_mutton_price.predict([[year, month, week]]),
                'Discount value' : 'prediction'
            }, index=[0])

            st.table(df)#,width = 10000)

        table()
