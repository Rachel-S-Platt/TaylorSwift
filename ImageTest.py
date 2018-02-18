from PIL import Image
import dlib
from skimage import io
from matplotlib import pyplot as plt
import cv2

def detect_faces(image):

    # print("adfasfasdfsfsa")
    #
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #
    # print(faces)
    # print("adfasfasdfsfsa")
    #
    # for (x,y,w,h) in faces:
    #     print("Values")
    #     print(x)
    #     print(y)
    #     print(w)
    #     print(h)

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    print(face_detector)
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    print(detected_faces)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    print(face_frames)

    return face_frames


def get_face(name):
    face = cv2.imread(name)
    b,g,r = cv2.split(face)           # get b, g, r
    face = cv2.merge([r,g,b])
    face2 = Image.fromarray(face)
    face2.save("Rachl.jpg")

    # print(face)
    detect = detect_faces(face)
    #
    print(detect)
    #
    # # for n, face_rect in enumerate(detect):
    # #     face = Image.fromarray(face).crop(face_rect)
    #
    face = Image.fromarray(face)
    face = face.crop(detect[0])
    #
    # face = face.convert("RGBA")
    #
    face = face.resize((140, 140), Image.BILINEAR)
    #
    img = Image.open('tyalor-tom.jpg')
    img_w, img_h = face.size
    offset = (img_w-20, img_h / 6)
    img.paste(face, offset)
    img.save("Test.jpg")

print("Hello")
get_face("Nash.JPG")
