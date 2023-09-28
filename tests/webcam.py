import cv2
import numpy as np
import os
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

def generateModel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    batch_size = 4
    guidance_scale = 3.0

    image = load_image("Capture.png")

    latents = sample_latents(
            batch_size = batch_size,
            model = model,
            diffusion = diffusion,
            guidance_scale = guidance_scale,
            model_kwargs = dict(images=[image] * batch_size),
            progress = True,
            clip_denoised = True,
            use_fp16 = True,
            use_karras = True,
            karras_steps = 64,
            sigma_min = 1e-3,
            sigma_max = 160,
            s_churn = 0,
    )

    render_mode = 'cuda'
    size = 64

    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        display(gif_widget(images))

def getColorMask(img):
    #lower = np.array([20, 100, 120])
    #upper = np.array([60, 130, 210])
    lower = np.array([35, 0, 91])
    upper = np.array([255, 51, 255])

    kernel = np.ones((5,5), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower, upper)
    #mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.dilate(mask, kernel, iterations=1)

    maskA = cv2.bitwise_not(mask)
    mask = cv2.bitwise_and(img, img, mask = maskA)

    return mask

    #return cv2.inRange(hsv, lower, upper)

torch.cuda.empty_cache()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()

    cv2.imshow('image', getColorMask(img))

    if cv2.waitKey(1) == ord('c'):
        print("Capture requested, writing image")
        #cv2.imwrite("Capture.png", getColorMask(img))
        generateModel()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
