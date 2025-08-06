import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.classifier.model import get_model
from src.classifier.utils import get_transforms

def run_inference(model_path, class_names, roi_box=(1200, 100, 700, 900), camera_index=0):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Transforms
    _, preprocess = get_transforms(input_size=224)

    # Webcam
    cap = cv2.VideoCapture(camera_index)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = roi_box
        roi = frame[y:y+h, x:x+w]

        # Preprocess
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_class = class_names[pred.item()]

        # Draw ROI + Prediction
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Pred: {pred_class}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display
        cv2.imshow("Webcam Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(project_root, "saved_models", "best_model.pth"))
    class_names = ['cardboard', 'glass', 'metal', 'plastic', 'trash']
    run_inference(model_path, class_names)
