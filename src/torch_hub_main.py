# pip3 install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies  
import torch
import cv2
import boto3
import numpy as np
from PIL import Image
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

def crop_image(frame, bboxes, filename, client=None):
    print("Cropping Image")

    for idx, bbox in enumerate(bboxes):
 
        int_bbox = [int(x) for x in bbox[:4]]
        x1,y1,x2,y2 = int_bbox
        crop = frame[y1:y2, x1:x2]
        
        # convert image to string to save to s3
        image_string = cv2.imencode('.jpg', crop)[1].tobytes()
        client.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'cropped/{filename}_{idx}.jpg')


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [14] # only return birds

    return(model)


if __name__=="__main__":
    load_dotenv()
    model = load_model()
    session = boto3.Session(
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    )
    s3 = session.client('s3')
    cam = cv2.VideoCapture(0)
    rest_end = datetime.now()
  
    while(True): 
        ret, frame = cam.read()
        frame = frame[:, :, [2,1,0]]
        frame = Image.fromarray(frame) 
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        result = model(frame,size=640)
        # cv2.imshow('Camera', frame)
        #cv2.imshow('YOLO', np.squeeze(result.render()))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break


        # x1, y1, x2, y2, confidence, class
        bboxes = result.xyxy[0].tolist() # img predictions (tensor) bounding box 
        if len(bboxes) > 0  and datetime.now() > rest_end:
            print("Found a Bird")
            result.print()

            
            filename = uuid.uuid4().hex

            # Crop image with bboxes 
            crop_image(frame, bboxes, filename, client=s3)
            # result.save(save_dir='images/bbox_images', exist_ok=True)  # save detection with bounding box

            image_string = cv2.imencode('.jpg', np.squeeze(result.render()))[1].tobytes()
            s3.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'bbox/{filename}.jpg')
            

            print("sleeping...")

            rest_end = datetime.now() + timedelta(seconds=15)
            

        

