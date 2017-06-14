import argparse
import glob
import sys
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
args = parser.parse_args()


def square(image):
    try:
        img = Image.open(image)
        width, height = img.size
        tam = height
        if (tam < width):
            tam = width
        size = (tam, tam)
        img = ImageOps.fit(img, size, Image.ANTIALIAS)
        img.save(image)
    except:
        print ("Unexpected error:" + str(sys.exc_info()[0]))
        pass


for image in glob.glob(args.dataFolder + "/*/*/*"):
    print(image)
    square(image)
