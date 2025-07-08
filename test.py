import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch.nn as nn

# Initialization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

# face recog model
face_rec_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Embedding for face user 
my_embedding = torch.load('my_face_embedding.pt').to(device)

#  anti-spoofing model (cnn model)
class AntiSpoofingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load('anti_spoofing_model.pth') 

    def forward(self, x):
        return self.model(x)

from torchvision import models
anti_spoof_model = models.resnet18(pretrained=False)
anti_spoof_model.fc = nn.Linear(anti_spoof_model.fc.in_features, 2)
anti_spoof_model.load_state_dict(torch.load('anti_spoofing_model.pth'))
anti_spoof_model = anti_spoof_model.to(device).eval()

# transformer similar to anti-spoofing
anti_spoof_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture vidéo")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # face detection
    face = mtcnn(img_pil)
    if face is not None:
        face_batch = face.unsqueeze(0).to(device)  

        # Anti-spoofing : crop face size 224x224 
        face_for_spoof = anti_spoof_transform(img_pil).unsqueeze(0).to(device)  

        #  anti-spoofing Prediction
        with torch.no_grad():
            spoof_outputs = anti_spoof_model(face_for_spoof)
            _, spoof_pred = torch.max(spoof_outputs, 1)
        
        if spoof_pred.item() == 0:  # 0 = réel, 1 = spoof 
            # facial recog
            with torch.no_grad():
                embedding = face_rec_model(face_batch).cpu()
                similarity = F.cosine_similarity(embedding, my_embedding.cpu())
                similarity_score = similarity.item()

            if similarity_score > 0.75:
                label = f"Access Granted ({similarity_score:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Access Denied ({similarity_score:.2f})"
                color = (255, 0, 0)
        else:
            label = "Spoof Detected"
            color = (0, 0, 255)
    else:
        label = "No face detected"
        color = (0, 0, 255)

    # frame label 
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Face Recognition + Anti Spoofing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
