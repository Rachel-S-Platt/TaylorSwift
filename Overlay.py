from PIL import Image
from skimage import io
# from matplotlib import pyplot as plt
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

def combine_faces(name1, name2):
    background = io.imread(name1)
    detect1 = detect_faces(background)

    for n, face_rect in enumerate(detect1):
        background = Image.fromarray(background).crop(face_rect)
        background.save("Test.png")

    overlay = io.imread(name2)
    detect2 = detect_faces(overlay)

    for n, face_rect in enumerate(detect2):
        overlay = Image.fromarray(overlay).crop(face_rect)

    # overlay2 = io.imread(name3)
    # detect3 = detect_faces(overlay2)
    #
    # for n, face_rect in enumerate(detect2):
    #     overlay2 = Image.fromarray(overlay2).crop(face_rect)


    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    # overlay2 = overlay2.convert("RGBA")

    background = background.resize((300, 300), Image.BILINEAR)
    overlay = overlay.resize((300, 300), Image.BILINEAR)
    # overlay2 = overlay2.resize((300, 300), Image.BILINEAR)

    new_img = Image.blend(background, overlay, 0.5)
    # new_img2 = Image.blend(new_img, overlay2, 0.5)
    new_img.save("new.png","PNG")

combine_faces("Rachel.jpg", "PJ.jpg")
