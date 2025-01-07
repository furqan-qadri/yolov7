import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import cv2
import numpy as np

# Load YOLOv7 model
weights = 'yolov7.pt'  # Path to the downloaded YOLOv7 weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
model = attempt_load(weights, map_location=device)  # Load the model
model.eval()  # Set model to evaluation mode

# Load class names (COCO dataset labels)
class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Function to perform object detection
def detect_objects(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img0 = img.copy()  # Keep the original for display
    img = letterbox(img, new_shape=(640, 640))[0]  # Resize with padding
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange dimensions
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device) / 255.0  # Normalize to 0-1
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)  # NMS

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  # Scale boxes to original image
            for *xyxy, conf, cls in det:
                label = f"{class_names[int(cls)]} {conf:.2f}"  # Add class name and confidence
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)


    # Display the image
    cv2.imshow('YOLOv7 Detection', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
image_path = "/Users/furqanqadri/Downloads/IMG_7911 4.JPG"  # Replace with the path to your image
detect_objects(image_path)
