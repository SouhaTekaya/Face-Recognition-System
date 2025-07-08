import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Detect & crop face
    face = mtcnn(img_pil)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = model(face)

        print("Face embedding vector:", embedding.shape)

        # Save embedding
        torch.save(embedding, 'my_face_embedding.pt')
        print("Embedding saved.")

        # Showing img
        plt.imshow(img_rgb)
        plt.title("Captured Face")
        plt.axis("off")
        plt.show()
    else:
        print("Aucun visage détecté.")
else:
    print("Impossible de lire depuis la caméra.")

cap.release()
