# Face-Recognition-System
This project is a real-time face recognition system developed using deep learning and computer vision techniques. It detects faces from live camera feeds, extracts facial features (embeddings), and recognizes individuals by comparing them to a known database. It also includes optional anti-spoofing (liveness detection) to enhance security against fake faces (photos/videos/masks).

To use it:

Prepare a dataset exemple : CASIA or your own , with real and spoof face images. You can use Roboflow to help organize and label the data.

Train the CNN model to classify real vs. spoof faces.

Run extract_embeddings.py to generate and save face embeddings for authorized users.

Finally, run test.py to launch the real-time system: it detects the face, checks for spoofing, and grants access if the face is real and recognized.

===================================
Before running the project, make sure to install the following Python libraries

pip install opencv-python
pip install numpy
pip install torch torchvision torchaudio
pip install facenet-pytorch
pip install matplotlib
pip install Pillow
pip install scikit-learn
pip install roboflow
pip install pypylon


