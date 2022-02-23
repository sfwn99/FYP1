from PIL import Image
import os

inPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\testimages"
outPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\editedimages"
imagePath = "cleanyellow.jpg"
# imagePath contains name of the image
inputPath = os.path.join(inPath, imagePath)

# inputPath contains the full directory name
img = Image.open(inputPath)

fullOutPath = os.path.join(outPath, 'invert_' + imagePath)
# fullOutPath contains the path of the output
# image that needs to be generated
img.thumbnail((256,256))
img.save(fullOutPath)

