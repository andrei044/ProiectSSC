import cv2
from matplotlib import pyplot as plt

CAMERA_INDEX=1


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

def main():
    record()
    # plt.imshow(frame)
    # plt.show()
    
if __name__ == "__main__":
    main()