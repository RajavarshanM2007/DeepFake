# my_models.py

import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO

def image_to_graph(image_bytes):
    """
    Simplified: convert image bytes to a normalized torch tensor
    (batch size 1, 3 x H x W), resizing to 64x64.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print("Error opening image:", e)
        return None
    image = image.resize((64, 64))
    arr = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1]
    # convert to torch tensor: shape [1, 3, 64, 64]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

def extract_faces_from_video(video_bytes, max_faces=5):
    """
    Simplified: placeholder. As extracting from video requires more heavy libs.
    We'll simulate by returning an empty list or maybe first frame.
    """
    # If you have time, you can decode the first frame using imageio or cv2
    try:
        import imageio
        reader = imageio.get_reader(video_bytes, "ffmpeg")
        for i, frame in enumerate(reader):
            if i == 0:
                # convert frame to RGB image array
                image_pil = Image.fromarray(frame).convert("RGB")
                image_pil = image_pil.resize((64, 64))
                return [image_pil]  # return list of one face image
            if i >= 1:
                break
    except Exception as e:
        print("Error in extract_faces_from_video:", e)
    return []
