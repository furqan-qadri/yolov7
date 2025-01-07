import cv2
import torch
import numpy as np

# YOLOv7 modules
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def detect_webcam(weights='yolov7.pt', conf_thres=0.25, iou_thres=0.45):
    """
    Opens the default webcam, runs YOLOv7 detection on each frame,
    and displays bounding boxes and labels. Press 'q' to exit.
    """
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLOv7 model
    model = attempt_load(weights, map_location=device)
    model.eval()  # set inference mode

    # Get class names directly from the model
    class_names = model.module.names if hasattr(model, 'module') else model.names

    # Open the webcam (device index 0)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Copy original frame to draw on
        img0 = frame.copy()

        # Preprocess: resize + pad to 640x640 while preserving aspect ratio
        resized, _, _ = letterbox(img0, new_shape=(640, 640))

        # BGR -> RGB, HWC -> CHW
        resized = resized[:, :, ::-1].transpose(2, 0, 1)
        # Convert to contiguous array for PyTorch
        resized = np.ascontiguousarray(resized)

        # Create a float32 torch tensor
        img_tensor = torch.from_numpy(resized).float().to(device)
        # Normalize 0-255 to 0-1
        img_tensor /= 255.0

        # Add batch dimension [1, 3, 640, 640]
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference without gradient tracking
        with torch.no_grad():
            preds = model(img_tensor)[0]               # Forward pass
            preds = non_max_suppression(preds, conf_thres, iou_thres)

        # Process detections
        for det in preds:
            if len(det):
                # Scale coords from resized shape back to original
                # Clone det[:, :4] to avoid issues with in-place ops
                scaled_coords = scale_coords(
                    img_tensor.shape[2:], 
                    det[:, :4].clone(),  # <-- clone() to avoid runtime error
                    img0.shape
                ).round()

                # Update detection boxes with scaled coords
                det[:, :4] = scaled_coords

                # Draw bounding boxes
                for *xyxy, conf, cls_id in det:
                    # Label text
                    label = f"{class_names[int(cls_id)]} {conf:.2f}"

                    # Draw rectangle
                    cv2.rectangle(
                        img0,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 255, 0),
                        2
                    )
                    # Put label text above box
                    cv2.putText(
                        img0,
                        label,
                        (int(xyxy[0]), int(xyxy[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # Show the result
        cv2.imshow('YOLOv7 Webcam Detection', img0)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_webcam()
