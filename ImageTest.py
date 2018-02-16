from PIL import Image

img = Image.open('tyalor-tom.jpg')
img_w, img_h = img.size
img2 = Image.open('Alex.jpg')
maxsize = (int(img2.size[0] * .25), int(img2.size[1] * .25))
print(maxsize)
img2 = img2.resize(maxsize)
offset = (img_w / 2, img_h / 2)
img.paste(img2, offset)
img.save("Test.jpg")
