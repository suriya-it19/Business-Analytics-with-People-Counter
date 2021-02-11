import Streamlit_app

#===============================================================================
""" Optional features config. """
#===============================================================================
PROTOTXT = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
MODEL = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

Skip_frames = 30
Confidence = Streamlit_app.conf#0.4
MAIL = ''
#url = 'http://192.168.1.2:8080/shot.jpg' url = 0
url1 = int(Streamlit_app.id1)
url2 = Streamlit_app.id2
url3 = Streamlit_app.id3

ALERT = False
Threshold = 10
Thread = False
Log = False
Scheduler = False
Timer = False
#===============================================================================
#===============================================================================
