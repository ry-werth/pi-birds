# pip3 install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies  
import torch
import cv2
import boto3
import numpy as np
from dotenv import load_dotenv
import os

def crop_image(image_file, bboxes, save_dir="images/cropped/", client=None):
    print("Cropping Image")
    cv_img= cv2.imread(image_file) 
    file_name, file_ext  =  image_file.split('/')[-1].split('.')

    path_list = []

    for idx, bbox in enumerate(bboxes):
 
        int_bbox = [int(x) for x in bbox[:4]]
        x1,y1,x2,y2 = int_bbox
        crop = cv_img[y1:y2, x1:x2]

        path = f'{save_dir}{file_name}_{idx}.{file_ext}'
        path_list.append(path)
        #cv2.imwrite(path, crop)
        
        # convert image to string to save to s3
        image_string = cv2.imencode('.jpg', crop)[1].tobytes()
        client.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'cropped/{file_name}_{idx}.jpg')

    return(path_list)

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [14] # only return birds

    return(model)


if __name__=="__main__":

    model = load_model()
    load_dotenv()
    session = boto3.Session(
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    )
    s3 = session.client('s3')

    # Images
    imgs = ['images/practice/bird_1.webp', 'images/practice/bird_2.jpeg', 'images/practice/bird_3.jpeg']  # batch of images
    img = 'images/practice/bird_3.jpeg'
    file_name_ext =  img.split('/')[-1]
    file_name, file_ext = file_name_ext.split('.')

    # Inference
    result = model(img)

    # x1, y1, x2, y2, confidence, class
    bboxes = result.xyxy[0].tolist() # img predictions (tensor) bounding box 
    if len(bboxes) > 0:
        print("Found a Bird")
        result.print()

        # Crop image with bboxes 
        crop_image(img, bboxes, save_dir="images/cropped/", client=s3)
        result.save(save_dir='images/bbox_images', exist_ok=True)  # save detection with bounding box

        image_string = cv2.imencode('.jpg', np.squeeze(result.render()))[1].tobytes()
        s3.put_object(Body=image_string, Bucket='birds-rywerth', Key=f'bbox/{file_name}.jpg')

    # TODO Save BBOX image somewhere and crop somewhwere. Nothing else. Right now crop also resaves original

