import zipfile, os, shutil, cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from mtcnn import MTCNN



def image_to_graph(image_tensor, k=9, patch_size=32, debug=True):
    """
    Converts an image tensor [3,H,W] into a graph where
    each node = flattened patch of size (3*patch_size*patch_size).
    """

    C, H, W = image_tensor.shape
    if H < patch_size or W < patch_size:
        if debug:
            print(f"⚠️ Skipping tiny frame: {image_tensor.shape}")
        return None

    # Extract patches
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()   # [num_h, num_w, C, ps, ps]
    patches = patches.view(-1, C * patch_size * patch_size)  # [num_patches, features]

    if patches.size(0) < 2:
        if debug:
            print(f"⚠️ Too few patches: {patches.shape}")
        return None

    # Normalize patches for cosine similarity
    patches_norm = F.normalize(patches, p=2, dim=1)  # [num_patches, features]

    # Compute similarity (matrix)
    similarity = torch.matmul(patches_norm, patches_norm.T)  # [num_patches, num_patches]

    # Build k-NN graph
    edge_index = []
    for i in range(similarity.size(0)):
        # Top-k neighbors (excluding self)
        k_eff = min(k + 1, similarity.size(1))  # don’t ask for more than available
        indices = torch.topk(similarity[i], k_eff).indices.tolist()
        indices = [j for j in indices if j != i]  # remove self
        edge_index += [(i, j) for j in indices]

    if len(edge_index) == 0:
        if debug:
            print("⚠️ No edges created")
        num_nodes = patches.size(0)
        edge_index = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = patches.float()  # node features
    batch = torch.zeros(x.size(0), dtype=torch.long)  # all belong to same graph

    return Data(x=x, edge_index=edge_index, batch=batch)


detector = MTCNN() #Initializes the MTCNN face detector
# 🔍 Face Extraction & Prediction Logic
def extract_faces_from_video(video_path, max_faces=10): #max faces it can analyse is 10
    cap = cv2.VideoCapture(video_path) #Uses OpenCV to open the video file frame by frame.
    faces = [] 
    count = 0
    while cap.isOpened() and count < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_faces(frame)
        for face in detections:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h, x:x+w] #Extracts the face region from the frame.
            try:
                face_img = cv2.resize(face_img, (64, 64))
                faces.append(face_img)
                count += 1
                break
            except:
                continue  # Skip bad crops

        if count >= max_faces:
            break
    cap.release()
    return faces