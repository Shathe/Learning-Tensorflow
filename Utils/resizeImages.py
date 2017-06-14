import PIL
import argparse
import glob
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
args = parser.parse_args()

def resize(image):
    try:
        img = Image.open(image)
        img = img.resize((int(args.width), int(args.height)), PIL.Image.ANTIALIAS)
        img.save(image)
    except:
        os.remove(image)


for image in glob.glob(args.dataFolder + "/*/*/*"):
    print(image)
    resize(image)

