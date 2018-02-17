from PIL import Image
import dlib
from skimage import io
from matplotlib import pyplot as plt

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames


def get_face(name):
    face = io.imread(name)
    detect = detect_faces(face)

    for n, face_rect in enumerate(detect):
        face = Image.fromarray(face).crop(face_rect)


    face = face.convert("RGBA")

    face = face.resize((140, 140), Image.BILINEAR)

    img = Image.open('tyalor-tom.jpg')
    img_w, img_h = face.size
    offset = (img_w-20, img_h / 6)
    img.paste(face, offset)
    img.save("Test.jpg")

get_face("Rachel.jpg")
