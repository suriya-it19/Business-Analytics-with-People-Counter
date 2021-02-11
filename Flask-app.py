import csv
import datetime
import time
import urllib.request
from itertools import zip_longest

import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
import schedule
import tensorflow
from flask import Flask, Response, render_template
from imutils.video import FPS, VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client

from mylib import config, thread
from mylib.centroidtracker import CentroidTracker
from mylib.mailer import Mailer
from mylib.trackableobject import TrackableObject

t0 = time.time()
app = Flask(__name__)

baseHtml = 'base.html'
indexHtml = 'index.html'

def gen_frames():

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	net = cv2.dnn.readNetFromCaffe(config.PROTOTXT, config.MODEL)

	print("[INFO] Starting the live stream..")
	vs = VideoStream(config.url).start()
	#vs = cv2.VideoCapture(config.url)
	time.sleep(2.0)

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

		if frame is None:
			break

		frame = imutils.resize(frame, width = 500)
		#frame = cv2.resize(frame, (500, 500))
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

		
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/webcam')
def cam_video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template(indexHtml)


if __name__ == '__main__':
	if config.Scheduler:
		schedule.every().day.at("9:00").do(app.run)

		while 1:
			schedule.run_pending()

	else:
		app.run()
