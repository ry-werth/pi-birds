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
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

def crop_image(image, bboxes, filename, client=None):
    print("Cropping Image")

    for idx, bbox in enumerate(bboxes):
 
        x1=bbox.origin_x
        y1=bbox.origin_y
        x2=bbox.origin_x + bbox.width
        y2=bbox.origin_y + bbox.height
        crop = image[y1:y2, x1:x2]

        # print(cv2.imwrite('images/crop/cropped.jpg',crop))
        
        # convert image to string to save to s3
        image_string = cv2.imencode('.jpg', crop)[1].tobytes()
        client.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'cropped/{filename}_{idx}.jpg')


if __name__=="__main__":
    load_dotenv()
    session = boto3.Session(
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    )
    s3 = session.client('s3')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name='models/efficientdet_lite0.tflite', use_coral=False, num_threads=4)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()
    rest_end = datetime.now()

    while(True): 
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )
        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        birds = [detection for detection in detection_result.detections if detection.categories[0].category_name  == 'bird']

        if len(birds) > 0  and datetime.now() > rest_end:
            print("Found a Bird")
            print(birds)
            # [Detection(bounding_box=BoundingBox(origin_x=246, origin_y=204, width=328, height=174), categories=[Category(index=15, score=0.7578125, display_name='', category_name='bird')])]

            filename = uuid.uuid4().hex

            for bird in birds:
                bbox = bird.bounding_box
                category = bird.categories[0]
                probability = round(category.score, 2)
                print(f"Probability: {probability}")

            bboxes = [bird.bounding_box for bird in birds]
            # Crop image with bboxes 
            crop_image(image, bboxes, filename, client=s3)
            # result.save(save_dir='images/bbox_images', exist_ok=True)  # save detection with bounding box

            # cv2.imwrite('images/orig/orig.jpg',image)
            image_string = cv2.imencode('.jpg', image)[1].tobytes()
            s3.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'original/{filename}.jpg')
            

            print("sleeping...")

            rest_end = datetime.now() + timedelta(seconds=15)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()
            print(fps)
            

        

