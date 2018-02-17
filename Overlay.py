from PIL import Image
from skimage import io
import dlib

# https://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def combine_faces(name1, name2, name3):
    layer1 = io.imread(name1)
    detect1 = detect_faces(layer1)

    for n, face_rect in enumerate(detect1):
        layer1 = Image.fromarray(layer1).crop(face_rect)
        layer1.save("Test.png")

    layer2 = io.imread(name2)
    detect2 = detect_faces(layer2)

    for n, face_rect in enumerate(detect2):
        layer2 = Image.fromarray(layer2).crop(face_rect)

    layer3 = io.imread(name3)
    detect3 = detect_faces(layer3)

    for n, face_rect in enumerate(detect3):
        layer3 = Image.fromarray(layer3).crop(face_rect)


    layer1 = layer1.convert("RGBA")
    layer2 = layer2.convert("RGBA")
    layer3 = layer3.convert("RGBA")

    layer1 = layer1.resize((300, 300), Image.BILINEAR)
    layer2 = layer2.resize((300, 300), Image.BILINEAR)
    layer3 = layer3.resize((300, 300), Image.BILINEAR)

    new_img = Image.blend(layer1, layer2, 0.5)
    new_img2 = Image.blend(new_img, layer3, 0.5)
    new_img2.save("Blend.png","PNG")

combine_faces("./static/img/Boyfriends/JohnMayer.jpg", "./static/img/Boyfriends/TaylorLautner.jpg", "./static/img/Boyfriends/JoeJonas.jpg")
