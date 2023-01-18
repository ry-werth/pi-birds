import torch
import cv2
import boto3
import numpy as np
from PIL import Image
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [14] # only return birds

    return(model)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)
frame_count = 0
start = time.time()
model = load_model()
while True:

    # read frame
    ret, frame = cap.read()
    frame = frame[:, :, [2,1,0]]
    frame = Image.fromarray(frame) 
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    result = model(frame,size=640)
    cv2.imshow('YOLO', np.squeeze(result.render()))
    #cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # calculate frame rate
    frame_count += 1
    if time.time() - start > 1:
        print(frame_count)
        frame_count = 0
        start = time.time()