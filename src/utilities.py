import cv2
from matplotlib import pyplot as plt

CAMERA_INDEX=1
IMAGE_WIDTH=800
IMAGE_HEIGHT=800


HAAR_CASCADE = 'cars.xml'

def take_photo():
    #Connect to webcam
    cap=cv2.VideoCapture(CAMERA_INDEX)
    #Read webcam input
    ret, frame=cap.read()

    cv2.imwrite('webcamphoto.jpg',frame)
    #Disconnect webcam
    cap.release()

def record():
    #Connect to webcam
    cap=cv2.VideoCapture(CAMERA_INDEX)
    #Loop while webcam is connected
    while cap.isOpened():
        ret, frame= cap.read()
        cv2.imshow('Webcam',frame)
        #When pressing q exit loop
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def record_detect():
    #Connect to webcam
    cap=cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    car_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    #Loop while webcam is connected
    while cap.isOpened():
        ret, frame= cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        for (x,y,w,h) in cars: 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) 
        cv2.imshow('Webcam',frame)
        #When pressing q exit loop
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()