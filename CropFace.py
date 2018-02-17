import dlib
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Load image
img_path = 'static/img/goat.jpeg'
image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
for n, face_rect in enumerate(detected_faces):
    face = Image.fromarray(image).crop(face_rect)
    face.save("Test.jpg")
#    plt.subplot(1, len(detected_faces), n+1)
#    plt.axis('off')
#    plt.imshow(face)
