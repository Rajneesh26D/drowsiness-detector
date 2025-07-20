import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from PIL import Image
from mtcnn import MTCNN

MODEL_PATH = "drowsiness_model.pth"
class_names = ['blink', 'none', 'yawn']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet18 model architecture with no pretrained weights
model = models.resnet18(weights=None)

# Freeze all layers to avoid updating during inference
for param in model.parameters():
    param.requires_grad = False

# Replace the final FC layer to match the 3 output classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 3)
)

# Load trained weights (saved with state_dict)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()  # Set model to evaluation mode

# Transformation to match the model input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

detector = MTCNN()  # detection using MTCNN

cap = cv2.VideoCapture(0)  # Start webcam feed
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)  # Detect face(s) in the current frame
    
    if faces:
        x, y, w, h = faces[0]['box']  # Extract coordinates of the first face
        x, y = max(0, x), max(0, y)   # no negative index
        
        face = rgb[y:y+h, x:x+w]  # Crop detected face region
        pil_img = Image.fromarray(face)  # Convert to PIL image for transform
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():  # Inference without gradient tracking
            output = model(input_tensor)
            _, pred = torch.max(output, 1)  # Get the index of the highest score
            label = class_names[pred.item()]  # Map index to class name

        # bounding box and label on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        #  alert text, when no face detected 
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection (Face Cropped)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
