import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time
import torch
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
import ttach as tta

from grad_cam import create_grad_cam


def our_image_classifier(image, model_plant_detector, model):
    '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
    m = nn.Sigmoid()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = {0:'Septoria' ,
         1:'Powdery Mildew',
         2:'Healthy',
         3:'Tobacco Mosiac Virus',
         4: 'Spider Mites',
         5:'Calcium Deficiency' ,
         6:'Magnesium Deficiency' }
    
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90,180]),
        ]
    )

    label_id, probabilitis, rect_image, HeatMap, grayscale_cam = None, None, None, None, None
    leaf_detected = False

    img = image.copy()

    transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(),
    ToTensorV2(),
    ])

    img = transform(image=img)["image"]
    
    img = img.unsqueeze(0)
    img_f = img.float()
    output = model_plant_detector(img_f.to(device))
    pred = m(output)
    pred = (np.array(pred.tolist()) > 0.5).astype(int).tolist()[0][0]
    
    if pred:
      st.success("Plant detected in image")
      leaf_detected = True

      tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

      model = model.eval()
      output = tta_model(img_f.to(device))
      rect_image, HeatMap, grayscale_cam = create_grad_cam(model, image, img)
    
      pred = m(output)
      ms = nn.Softmax()
      probabilitis = ms(output)

      pred_max = torch.argmax(pred, dim = 1)
      label_id = pred_max.tolist()[0]
      label = labels[label_id]
    else:
      st.warning("No Plant detected in image")

    return label_id, probabilitis, rect_image, HeatMap, grayscale_cam, leaf_detected
