import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import cv2
import numpy as np

# Load YOLOv7 model and COCO class names
weights = 'yolov7.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)
model.eval()
class_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(image_path):
    img = cv2.imread(image_path)
    img0 = img.copy()
    img = letterbox(img, new_shape=(640, 640))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(np.ascontiguousarray(img)).float().to(device) / 255.0
    img = img.unsqueeze(0) if img.ndimension() == 3 else img

    with torch.no_grad():
        pred = non_max_suppression(model(img)[0], 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{class_names[int(cls)]} {conf:.2f}"
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow('YOLOv7 Detection', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
detect_objects("/Users/furqanqadri/Downloads/IMG_7911 4.JPG")
