import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def create_grad_cam(model, image, input_tensor):

    target_layers = [model.encoder.layers.encoder_layer_11.ln_1]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None,aug_smooth=True, eigen_smooth=True)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    image = cv2.resize(image, (224, 224))
    HeatMap = show_cam_on_image(image.astype(np.float32)/255, grayscale_cam, use_rgb=True)

    pred_map = grayscale_cam > 0.55
    contours = cv2.findContours(pred_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    rect_image = cv2.drawContours(image, contours, -1, (255,0,0),1)

    return rect_image, HeatMap, grayscale_cam
