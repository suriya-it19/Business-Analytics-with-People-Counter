#===============================================================================
""" Optional features config. """
#===============================================================================
PROTOTXT = 'People_counter_with_analytics/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
MODEL = 'People_counter_with_analytics/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

Skip_frames = 30
Confidence = 0.4
# Enter mail below to receive real-time email alerts
# e.g., 'email@gmail.com'
MAIL = ''
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
#url = 'http://192.168.1.2:8080/shot.jpg'
url = 0

# ON/OFF for mail feature. Enter True to turn on the email alert feature.
ALERT = False
# Set max. people inside limit. Optimise number below: 10, 50, 100, etc.
Threshold = 10
# Threading ON/OFF
Thread = False
# Simple log to log the counting data
Log = False
# Auto run/Schedule the software to run at your desired time
Scheduler = False
# Auto stop the software after certain a time/hours
Timer = False
#===============================================================================
#===============================================================================
