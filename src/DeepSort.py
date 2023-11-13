import cv2
import numpy as np
import sys
import glob
import time
import torch

class YoloDetector():
    def __init__(self,modelName):
        self.model=self.load_model(modelName)
        self.classes=self.model.names
        #print(self.classes)
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ",self.device)

    def load_model(self,modelName):
        model=torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
        return model

    def score_frame(self,frame):
        self.model.to(self.device)
        downscale_factor=2
        width=int(frame.shape[1] / downscale_factor)
        height=int(frame.shape[0] / downscale_factor)
        frame=cv2.resize(frame,(width,height))
        #frame=frame.to(self.device)

        results=self.model(frame)
        labels,cord=results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]

        return labels,cord

    def class_to_label(self,x):
        return self.classes[int(x)]
    
    def plot_boxes(self,results,frame,height,width,classLabel,confidence=0.3):

        labels,cord=results
        detections=[]

        n=len(labels)
        x_shape, y_shape= width,height
        
        for i in range(n):
            row=cord[i]
            if row[4]>=confidence:
                x1,y1,x2,y2 = int(row[0]*x_shape), int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)

                if self.class_to_label(labels[i])==classLabel:
                    # x_center=x1+((x2-x1)/2)
                    # y_center=y1+((y2-y1)/2)
                    # tlwh=np.asarray([x1,y1,int(x2-x1),int(y2-y1)], dtype=np.float32)
                    confidence=float(row[4].item())
                    # feature='cell phone'

                    detections.append(([x1,y1,int(x2-x1),int(y2-y1)],confidence,classLabel))
        return frame, detections
    

CAMERA_INDEX=1
IMAGE_WIDTH=800
IMAGE_HEIGHT=800
cap=cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

detector=YoloDetector(modelName=None)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from deep_sort_realtime.deepsort_tracker import DeepSort

object_tracker=DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

while cap.isOpened():
    success,img=cap.read()
    start=time.perf_counter()

    results=detector.score_frame(img)
    img,detections=detector.plot_boxes(results,img,height=img.shape[0],width=img.shape[1],classLabel='cell phone')

    tracks=object_tracker.update_tracks(detections,frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id=track.track_id
        ltrb=track.to_ltrb()

        bbox=ltrb

        cv2.rectangle(img,(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])),(0,0,255),2)
        if track.det_conf!= None:
            cv2.putText(img,"Id: "+str(track_id)+' Class: '+track.det_class+' Conf: '+str(round(track.det_conf,2)),(int(bbox[0]),int(bbox[1]-10)),cv2.FONT_ITALIC,0.3,(50, 168, 111),2)
        else:
            cv2.putText(img,"Id: "+str(track_id)+' Class: '+track.det_class+' Conf: None',(int(bbox[0]),int(bbox[1]-10)),cv2.FONT_ITALIC,0.3,(50, 168, 111),2)
            

    end=time.perf_counter()
    totalTime=end-start
    fps=1/totalTime

    cv2.putText(img,f'FPS: {round(fps,2)}',(20,70),cv2.FONT_ITALIC,1,(50, 168, 111),2)
    cv2.imshow('image',img)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()